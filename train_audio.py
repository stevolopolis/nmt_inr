import os
import datetime
import yaml
import shutil
import torch
import numpy as np
import hydra
import logging
import wandb

from omegaconf import OmegaConf
from easydict import EasyDict
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_func

from nmt import NMT
from utils import seed_everything, get_dataset, get_model, prep_audio_for_eval


log = logging.getLogger(__name__)


def load_config(config_file):
    configs = yaml.safe_load(open(config_file))
    return configs


def save_src_for_reproduce(configs, out_dir):
    if os.path.exists(os.path.join('outputs', out_dir, 'src')):
        shutil.rmtree(os.path.join('outputs', out_dir, 'src'))
    shutil.copytree('models', os.path.join('outputs', out_dir, 'src', 'models'))
    # dump config to yaml file
    OmegaConf.save(dict(configs), os.path.join('outputs', out_dir, 'src', 'config.yaml'))


def train(configs, model, dataset, device='cuda'):
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    exp_configs = configs.EXP_CONFIGS
    network_configs = configs.NETWORK_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # optimizer and scheduler
    if exp_configs.optimizer_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=network_configs.lr)
    elif exp_configs.optimizer_type == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=network_configs.lr)
    if exp_configs.lr_scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=0)
    elif exp_configs.lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)

    # prep model for training
    model.train()
    model = model.to(device)

    # prepare training settings
    process_bar = tqdm(range(train_configs.iterations))
    T, C = dataset.T, dataset.C
    best_psnr = 0
    best_pred = None

    # get data 
    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)
    ori_audio = labels.flatten().cpu().detach().numpy()
    ori_audio = (ori_audio*2) - 1 if model_configs.INPUT_OUTPUT.data_range == 0 else ori_audio # [-1,1]

    # nmt setup
    nmt = NMT(model,
              train_configs.iterations,
              (T, C),
              exp_configs.scheduler_type,
              exp_configs.strategy_type,
              exp_configs.mt_ratio,
              exp_configs.top_k,
              save_samples_path=train_configs.sampling_path,
              save_losses_path=train_configs.loss_path,
              save_name=None,
              save_interval=train_configs.save_interval)
    

    # sampling log
    sampling_history = dict()
    loss_history = dict()
    psnr_milestone = False

    # train
    for step in process_bar:
        # mt sampling
        sampled_coords, sampled_labels, full_preds = nmt.sample(step, coords, labels)

        sampled_preds = model(sampled_coords, None)
        # MSE loss
        loss = ((sampled_preds - sampled_labels) ** 2).mean()

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # process reconstructed audio for evaluation
        preds = prep_audio_for_eval(full_preds, model_configs, T, C)

        # evaluation
        psnr_score = psnr_func(preds, ori_audio, data_range=2)

        # W&B logging
        if configs.WANDB_CONFIGS.use_wandb:
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "lr": scheduler.get_last_lr()[0],
                        "mt": nmt.get_ratio(),
                        "mt_interval": nmt.get_interval()
                        }
            # Save ground truth image (only at 1st iteration)
            if step == 0:
                log_dict["GT"] =  wandb.Audio(ori_audio, sample_rate=16000)
                
            # Save reconstructed audio
            if step%train_configs.save_interval==0:
                log_dict["Reconstruction"] =  wandb.Audio(preds, sample_rate=16000)

            if not psnr_milestone and psnr_score > 40:
                psnr_milestone = True
                wandb.log({"PSNR Threshold": step}, step=step)

            wandb.log(log_dict, step=step)

        # Save model weights if it has the best PSNR so far
        if psnr_score > best_psnr:
            best_psnr = psnr_score
            best_pred = preds
            # torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, loss: {loss.item():.4f}")
    
    # wrap up training
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}")
    # W&B logging of final step
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.log(
                {
                "best_psnr": best_psnr,
                "best_pred": wandb.Audio(best_pred, sample_rate=16000),
                }, 
            step=step)
        wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}")
    
    # save model
    # torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

    return best_psnr


@hydra.main(version_base=None, config_path='config', config_name='train_audio')
def main(configs):
    configs = EasyDict(configs)

    # Save run name with current time
    # time_str = str(datetime.datetime.now().time()).replace(":", "").replace(".", "")
    # configs.TRAIN_CONFIGS.out_dir += "_" + time_str

    # Seed python, numpy, pytorch
    seed_everything(configs.TRAIN_CONFIGS.seed)
    # Saving config and settings for reproduction
    save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)

    # model configs
    configs.model_config.INPUT_OUTPUT.data_range = configs.NETWORK_CONFIGS.data_range
    configs.model_config.INPUT_OUTPUT.coord_mode = configs.NETWORK_CONFIGS.coord_mode
    if configs.model_config.name == "FFN":
        configs.model_config.NET.rff_std = configs.NETWORK_CONFIGS.rff_std

    # model and dataloader
    dataset = get_dataset(configs.DATASET_CONFIGS, configs.model_config.INPUT_OUTPUT)
    model = get_model(configs.model_config, dataset)
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")
    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"No. of parameters: {n_params}")
    
    # wandb
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.init(
            project=configs.WANDB_CONFIGS.wandb_project,
            entity=configs.WANDB_CONFIGS.wandb_entity,
            config=configs,
            group=configs.WANDB_CONFIGS.group,
            name=configs.TRAIN_CONFIGS.out_dir,
        )

        wandb.run.summary['n_params'] = n_params

    # train
    psnr = train(configs, model, dataset, device=configs.TRAIN_CONFIGS.device)

    return psnr

if __name__=='__main__':
    main()