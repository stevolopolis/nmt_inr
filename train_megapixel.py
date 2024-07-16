import os
import yaml
import shutil
import torch
import numpy as np
import hydra
import logging
from omegaconf import OmegaConf
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image
from utils import seed_everything, get_dataset, get_model, prep_image_for_eval, save_image_to_wandb
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import wandb

import datetime

from nmt import NMT

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

    # prepare training settings
    model.train()
    model = model.to(device)
    process_bar = tqdm(range(train_configs.iterations))
    H, W, C = dataset.H, dataset.W, dataset.C
    best_psnr, best_ssim = 0, 0
    best_pred = None
    
    # nmt setup
    nmt = NMT(model,
              train_configs.iterations,
              (H, W, C),
              exp_configs.scheduler_type,
              exp_configs.strategy_type,
              exp_configs.mt_ratio,
              exp_configs.top_k,
              save_samples_path=None,
              save_losses_path=None,
              save_name=None,
              save_interval=train_configs.save_interval)

    # placeholders for the original image and reconstructed image
    # we need these because we sample the megapixel in batches (not in a whole)
    ori_img = np.zeros(dataset.get_data_shape())
    ori_img_pred = np.zeros(dataset.get_data_shape())

    # train
    for step in process_bar:
        # run batch-wise mt sampling on entire data without updating model weights first
        # then, re-inference and do backpropagation
        sampled_coords_arr, sampled_labels_arr = [], []
        full_preds_arr = []
        iter_dataset = iter(dataset)
        for _ in range(len(dataset)):
            coords, labels = next(iter_dataset)
            coords, labels = coords.to(device), labels.to(device)

            # mt sampling
            sampled_coords, sampled_labels, full_preds = nmt.sample(step, coords, labels)
            # save the sampling results in a list
            sampled_coords_arr.append(sampled_coords)
            sampled_labels_arr.append(sampled_labels)
            full_preds_arr.append(full_preds)

        # backpropagation pipeline
        iter_dataset = iter(dataset)
        for batch in range(len(dataset)):
            coords, labels = next(iter_dataset)
            sampled_coords, sampled_labels = sampled_coords_arr[batch], sampled_labels_arr[batch]
            sampled_coords, sampled_labels = sampled_coords.to(device), sampled_labels.to(device)
            full_preds = full_preds_arr[batch]

            # inference
            sampled_preds = model(sampled_coords, None) 

            # MSE loss
            loss = ((sampled_preds - sampled_labels) ** 2).mean()

            # backprop
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()

            # rescale coords back to original image size
            if model_configs.INPUT_OUTPUT.coord_mode != 0:
                if model_configs.INPUT_OUTPUT.coord_mode != 1:
                    coords = (coords + 1) / 2
                coords = (coords * torch.tensor([W, H]).to(coords.device))

            # process reconstructed image for evaluation
            full_preds = prep_image_for_eval(full_preds, model_configs, H, W, C, reshape=False)

            # fill in the original image and reconstructed image with this batch
            x0_coord = torch.round(coords[:, 0].cpu()).long().clamp(0, W-1)
            x1_coord = torch.round(coords[:, 1].cpu()).long().clamp(0, H-1)
            ori_img[x0_coord, x1_coord] = labels.detach().cpu().numpy()
            ori_img_pred[x0_coord, x1_coord] = full_preds

            # log the step loss to wandb
            if configs.WANDB_CONFIGS.use_wandb:
                log_dict = {"step_loss": loss.item()}
                wandb.log(log_dict, step=step*len(dataset)+batch)
        
        # squeeze the image if image is GRAYSCALE
        if ori_img_pred.shape[-1] == 1:
            ori_img_pred = ori_img_pred.squeeze(-1) # Grayscale image

        # step the lr scheduler
        scheduler.step()

        # W&B logging
        if configs.WANDB_CONFIGS.use_wandb and step%train_configs.save_interval==0:
            # eval the reconstructed image
            psnr_score = psnr_func(ori_img_pred, ori_img, data_range=1)
            ssim_score = ssim_func(ori_img_pred, ori_img, channel_axis=-1, data_range=1)
            # log the eval stats
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "ssim": ssim_score,
                        "lr": scheduler.get_last_lr()[0],
                        "mt": nmt.get_ratio(),
                        "mt_interval": nmt.get_interval()
                        }

            # save reconstructed image
            save_image_to_wandb(log_dict, ori_img_pred, "Reconstruction", dataset_configs, H, W)

            # log to wandb
            wandb.log(log_dict, step=step)

        # check if the current psnr score is the highest
        if psnr_score > best_psnr:
            # update psnr and ssim score
            best_psnr, best_ssim = psnr_score, ssim_score
            # update best reconstructed image
            best_pred = ori_img_pred
            # Save best full resolution image locally
            predicted_img = Image.fromarray((ori_img_pred * 255).astype(np.uint8), mode=dataset_configs.color_mode)
            predicted_img.save(os.path.join('outputs', out_dir, 'best_pred.png'))

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}")
        
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    if configs.WANDB_CONFIGS.use_wandb:
        # log the highest psnr and ssim score to wandb
        best_pred = best_pred.squeeze(-1) if dataset_configs.color_mode == 'L' else best_pred
        wandb.log(
                {
                "best_psnr": best_psnr,
                "best_ssim": best_ssim
                }, 
            step=step)
        
        # save the best reconstruction to wandb before ending the run
        save_image_to_wandb(log_dict, best_pred, "best_pred", dataset_configs, H, W)
        wandb.finish()
    log.info(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))
    return best_psnr, best_ssim


@hydra.main(version_base=None, config_path='config', config_name='train_megapixel')
def main(configs):
    configs = EasyDict(configs)

    # Save run name with current time
    time_str = str(datetime.datetime.now().time()).replace(":", "").replace(".", "")
    configs.TRAIN_CONFIGS.out_dir += "_" + time_str

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
    configs.model_config.NET.num_layers = configs.NETWORK_CONFIGS.num_layers
    configs.model_config.NET.dim_hidden = configs.NETWORK_CONFIGS.dim_hidden
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