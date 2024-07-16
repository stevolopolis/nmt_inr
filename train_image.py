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

from nmt import NMT
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
    # load configs
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

    # nmt setup
    nmt = NMT(model,
              train_configs.iterations,
              (H, W, C),
              exp_configs.scheduler_type,
              exp_configs.strategy_type,
              exp_configs.mt_ratio,
              exp_configs.top_k,
              save_samples_path=train_configs.sampling_path,
              save_losses_path=train_configs.loss_path,
              save_name=None,
              save_interval=train_configs.save_interval)
    
    # sampling log
    psnr_milestone = False

    # train
    for step in process_bar:
        # mt sampling
        sampled_coords, sampled_labels, full_preds = nmt.sample(step, coords, labels)

        # subset inference for backprop
        sampled_preds = model(sampled_coords, None) 
        
        # MSE loss
        loss = ((sampled_preds - sampled_labels) ** 2).mean()

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # eval reconstruction (only if no_io is False)
        # if no_io is True, we skip all eval and logging to get the raw training speed
        psnr_score = 0
        ssim_score = 0
        if not train_configs.no_io:
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
                        "mt": nmt.get_ratio(),
                        "mt_interval": nmt.get_interval()
                        }
            # Save ground truth image (only at 1st iteration)
            if step == 0 and not train_configs.no_io:
                save_image_to_wandb(log_dict, ori_img, "GT", dataset_configs, H, W)
                
            # Save reconstructed image (and visualize sampled points)
            if step%train_configs.save_interval==0 and not train_configs.no_io:
                # save the reconstruction image
                save_image_to_wandb(log_dict, preds, "Reconstruction", dataset_configs, H, W)
                # load saved tinted image
                tinted_img = np.asarray(Image.open(nmt.get_saved_tint_path()))
                tinted_img = tinted_img / 255
                # save the tinted image
                save_image_to_wandb(log_dict, tinted_img, "Sampled points", dataset_configs, H, W)

            # if PSNR > 30, log the step
            if not psnr_milestone and psnr_score > 30:
                psnr_milestone = True
                wandb.log({"PSNR Threshold": step}, step=step)

            # log to wandb
            wandb.log(log_dict, step=step)

        # Save model weights if it has the best PSNR so far
        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_pred = preds
            if not train_configs.no_io:
                torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}, loss: {loss.item():.4f}")
    
    # wrap up training
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    # W&B logging of final step
    if configs.WANDB_CONFIGS.use_wandb:
        # if no_io is False, we log best-psnr, best-ssim, and the best prediction image
        if not train_configs.no_io:
            best_pred = best_pred.squeeze(-1) if dataset_configs.color_mode == 'L' else best_pred
            wandb.log(
                    {
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "best_pred": wandb.Image(Image.fromarray((best_pred*255).astype(np.uint8), mode=dataset_configs.color_mode)),
                    }, 
                step=step)
        # if no_io is True, we only log the best-psnr and best-ssim
        else:
            wandb.log(
                    {
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim
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
    if hasattr(configs.NETWORK_CONFIGS, "num_layers"):
        configs.model_config.NET.num_layers = configs.NETWORK_CONFIGS.num_layers
    if hasattr(configs.NETWORK_CONFIGS, "dim_hidden"):
        configs.model_config.NET.dim_hidden = configs.NETWORK_CONFIGS.dim_hidden

    # model and dataloader
    print(configs.DATASET_CONFIGS)
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