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

from nmt.scheduler import *
from nmt.sampler import mt_sampler, save_samples, save_losses
from nmt.strategy import strategy_factory, incremental, exponential


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

    # prepare training settings
    model.train()
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    process_bar = tqdm(range(train_configs.iterations))
    H, W, C = dataset.H, dataset.W, dataset.C
    best_psnr, best_ssim = 0, 0
    best_pred = None

    ori_img = np.zeros(dataset.get_data_shape())
    ori_img_pred = np.zeros(dataset.get_data_shape())

    sampling_history = dict()
    loss_history = dict()
    psnr_milestone = False

    # train
    for step in process_bar:
        iter_dataset = iter(dataset)
        # mt sampling
        mt_ratio = mt_scheduler(step, train_configs.iterations, float(exp_configs.mt_ratio))
        mt, mt_intervals = strategy(step, train_configs.iterations)

        if mt: 
            sampled_coords_arr, sampled_labels_arr = [], []
            full_preds_arr = []
            for _ in range(len(dataset)):
                coords, labels = next(iter_dataset)
                coords, labels = coords.to(device), labels.to(device)
            
                with torch.no_grad():
                    if exp_configs.top_k:
                        full_preds = model(coords, None)
                    else: 
                        full_preds = None
                    sampled_coords_iter, sampled_labels_iter, idx, dif = mt_sampler(coords, labels, full_preds, mt_ratio, top_k=exp_configs.top_k)
                    #tinted_labels = tint_data_with_samples(labels, idx, model_configs)
                    #if step % train_configs.mt_save_interval == 0:
                    #    save_samples(sampling_history, step, train_configs.iterations, sampled_coords, train_configs.sampling_path)
                    #    save_losses(loss_history, step, train_configs.iterations, dif, train_configs.loss_path)

                sampled_coords_arr.append(sampled_coords_iter)
                sampled_labels_arr.append(sampled_labels_iter)
                full_preds_arr.append(full_preds)

        iter_dataset = iter(dataset)
        for batch in range(len(dataset)):
            coords, labels = next(iter_dataset)
            if not mt and mt_intervals is None:
                sampled_coords, sampled_labels = coords, labels
            else:
                sampled_coords, sampled_labels = sampled_coords_arr[batch], sampled_labels_arr[batch]
            sampled_coords, sampled_labels = sampled_coords.to(device), sampled_labels.to(device)

            sampled_preds = model(sampled_coords, None) 
            if not mt and mt_intervals is None:
                full_preds = sampled_preds
            else:
                full_preds = full_preds_arr[batch]

            # MSE loss
            loss = ((sampled_preds - sampled_labels) ** 2).mean()

            if batch < len(dataset) - dataset.get_test_idx():
                # backprop
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                opt.step()
            
            # process reconstructed image for evaluation
            preds = prep_image_for_eval(full_preds, model_configs, H, W, C, reshape=False)

            if model_configs.INPUT_OUTPUT.coord_mode != 0:
                if model_configs.INPUT_OUTPUT.coord_mode != 1:
                    coords = (coords + 1) / 2
                coords = (coords * torch.tensor([W, H]).to(coords.device))

            x0_coord = torch.round(coords[:, 0].cpu()).long().clamp(0, W-1)
            x1_coord = torch.round(coords[:, 1].cpu()).long().clamp(0, H-1)
            ori_img[x0_coord, x1_coord] = labels.detach().cpu()
            ori_img_pred[x0_coord, x1_coord] = preds

            if configs.WANDB_CONFIGS.use_wandb:
                log_dict = {"loss": loss.item()}
                wandb.log(log_dict, step=step)
        
        if ori_img_pred.shape[-1] == 1:
            ori_img_pred = ori_img_pred.squeeze(-1) # Grayscale image

        scheduler.step()

        if configs.WANDB_CONFIGS.use_wandb and step%train_configs.save_interval==0:
            psnr_score = psnr_func(ori_img_pred, ori_img, data_range=1)
            ssim_score = ssim_func(ori_img_pred, ori_img, channel_axis=-1, data_range=1)
            log_dict = {
                        "loss": loss.item(),
                        "psnr": psnr_score,
                        "ssim": ssim_score,
                        "lr": scheduler.get_last_lr()[0],
                        "mt": mt_ratio,
                        "mt_interval": mt_intervals
                        }
            
            save_image_to_wandb(log_dict, ori_img_pred, "Reconstruction", dataset_configs, H, W)

            wandb.log(log_dict, step=step)

        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_pred = ori_img_pred
            # Save best full resolution image
            predicted_img = Image.fromarray((ori_img_pred * 255).astype(np.uint8), mode=dataset_configs.color_mode)
            predicted_img.save(os.path.join('outputs', out_dir, 'best_pred.png'))

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}")
        
    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    if configs.WANDB_CONFIGS.use_wandb:
        best_pred = best_pred.squeeze(-1) if dataset_configs.color_mode == 'L' else best_pred
        wandb.log(
                {
                "best_psnr": best_psnr,
                "best_ssim": best_ssim
                }, 
            step=step)
            
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