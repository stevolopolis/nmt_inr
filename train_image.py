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
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

from scheduler import *
from sampler import mt_sampler, save_samples, save_losses
from strategy import strategy_factory, incremental, exponential
from utils import seed_everything, get_dataset, get_model, prep_image_for_eval, save_image_to_wandb


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


def tint_data_with_samples(data, sample_idx, model_configs):
    """Relabel the data with given vis_label at the sample_idx indices."""
    if sample_idx is None: 
        return None
    
    new_data = data.detach().clone()
    if model_configs.INPUT_OUTPUT.data_range == 1:
        vis_label = torch.tensor([0.75, 0.0, 0.0]).to(data.device)
    else:
        vis_label = torch.tensor([0.5, 0.0, 0.0]).to(data.device)
    if data.shape[-1] == 1:
        vis_label = vis_label[0]

    new_data[sample_idx] = torch.clamp(new_data[sample_idx] + vis_label, max=1.0)

    return new_data


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
        opt = torch.optim.SGD(model.parameters(), lr=network_configs.lr, momentum=0.9)
    if exp_configs.lr_scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=0)
    elif exp_configs.lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)

    mt_scheduler = mt_scheduler_factory(exp_configs.scheduler_type)
    strategy = strategy_factory(exp_configs.strategy_type)

    # prep model for training
    model.train()
    model = model.to(device)

    # prepare training settings
    process_bar = tqdm(range(train_configs.iterations), disable=True)
    # process_bar = tqdm(range(1000)) # TODO: for sweep only
    H, W, C = dataset.H, dataset.W, dataset.C
    best_psnr, best_ssim = 0, 0
    best_pred = None

    # get data 
    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)
    # process training labels into ground truth image (for later use)
    ori_img = labels.view(H, W, C).cpu().detach().numpy()
    ori_img = (ori_img + 1) / 2 if model_configs.INPUT_OUTPUT.data_range == 1 else ori_img
    
    # sampling log
    sampling_history = dict()
    loss_history = dict()
    psnr_milestone = False

    # train
    for step in process_bar:
        # mt sampling
        mt_ratio = mt_scheduler(step, train_configs.iterations, float(exp_configs.mt_ratio))
        mt, mt_intervals = strategy(step, train_configs.iterations)
        if mt:
            with torch.no_grad():
                full_preds = model(coords, None)
                sampled_coords, sampled_labels, idx, dif = mt_sampler(coords, labels, full_preds, mt_ratio, top_k=exp_configs.top_k)
                tinted_labels = tint_data_with_samples(labels, idx, model_configs)
                if step % train_configs.mt_save_interval == 0:
                    save_samples(sampling_history, step, train_configs.iterations, sampled_coords, f"{train_configs.sampling_path}")
                    save_losses(loss_history, step, train_configs.iterations, dif, f"{train_configs.loss_path}")
        elif not mt and mt_intervals is None:
            sampled_coords, sampled_labels = coords, labels
            tinted_labels = None

        sampled_preds = model(sampled_coords, None) 
        if not mt and mt_intervals is None:
            full_preds = sampled_preds
        
        # MSE loss
        loss = ((sampled_preds - sampled_labels) ** 2).mean()

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # process reconstructed image for evaluation
        preds = prep_image_for_eval(full_preds, model_configs, H, W, C)

        # evaluation
        psnr_score = psnr_func(preds, ori_img, data_range=1)
        ssim_score = ssim_func(preds, ori_img, channel_axis=-1, data_range=1)

        # (optional) squeeze image if it is GRAYSCALE
        if preds.shape[-1] == 1:
            preds = preds.squeeze(-1)

        # W&B logging
        if configs.WANDB_CONFIGS.use_wandb:
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "ssim": ssim_score,
                        "lr": scheduler.get_last_lr()[0],
                        "mt": mt_ratio,
                        "mt_interval": mt_intervals
                        }
            # Save ground truth image (only at 1st iteration)
            if step == 0:
                save_image_to_wandb(log_dict, ori_img, "GT", dataset_configs, H, W)
                
            # Save reconstructed image (and visualize sampled points)
            if step%train_configs.save_interval==0:
                save_image_to_wandb(log_dict, preds, "Reconstruction", dataset_configs, H, W)
                if tinted_labels is not None:
                    tinted_img = prep_image_for_eval(tinted_labels, model_configs, H, W, C)
                    save_image_to_wandb(log_dict, tinted_img, "Sampled points", dataset_configs, H, W)

            if not psnr_milestone and psnr_score > 30:
                psnr_milestone = True
                wandb.log({"PSNR Threshold": step}, step=step)

            wandb.log(log_dict, step=step)

        # Save model weights if it has the best PSNR so far
        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_pred = preds
            torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}, loss: {loss.item():.4f}")
    
    # wrap up training
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    # W&B logging of final step
    if configs.WANDB_CONFIGS.use_wandb:
        best_pred = best_pred.squeeze(-1) if best_pred.shape[-1] == 1 else best_pred
        wandb.log(
                {
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "best_pred": wandb.Image(Image.fromarray((best_pred*255).astype(np.uint8), mode=dataset_configs.color_mode)),
                }, 
            step=step)
        wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

    return best_psnr, best_ssim


@hydra.main(version_base=None, config_path='config', config_name='train_image')
def main(configs):
    configs = EasyDict(configs)

    # Save run name with configs
    img_name = configs.DATASET_CONFIGS.file_path.split("/")[-1].split(".")[0]
    configs.TRAIN_CONFIGS.out_dir = f"{configs.EXP_CONFIGS.optimizer_type}_{configs.model_config.name}_mt{configs.EXP_CONFIGS.mt_ratio}_{configs.EXP_CONFIGS.strategy_type}_{configs.EXP_CONFIGS.scheduler_type}_{configs.EXP_CONFIGS.lr_scheduler_type}_topk{configs.EXP_CONFIGS.top_k}_{img_name}"

    # Seed python, numpy, pytorch
    seed_everything(configs.TRAIN_CONFIGS.seed)
    # Saving config and settings for reproduction
    save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)

    #########################
    # Specified because same config file was used to train
    # 8-layer SIRENS.
    # Only temporary.
    #########################
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
    psnr, ssim = train(configs, model, dataset, device=configs.TRAIN_CONFIGS.device)

    return psnr

if __name__=='__main__':
    main()