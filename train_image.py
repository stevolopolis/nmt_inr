import os
import datetime
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


def mt_sampler(data, y, preds, size):
    # Given size is ratio of training data
    if type(size) == float:
        n = int(size * len(data))
    # Given size is actual number of training data
    else:
        n = int(size)

    # mt sampling (returns indices)
    dif = torch.sum(torch.abs(y-preds), 1)
    _, ind = torch.topk(dif, n)

    # get sampled data
    sampled_data = data[ind]
    sampled_y = y[ind]
    
    return sampled_data, sampled_y


def train(configs, model, dataset, device='cuda'):
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    model_configs = configs.model_config
    out_dir = train_configs.out_dir

    # optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)

    # prep model for training
    model.train()
    model = model.to(device)

    # prepare training settings
    process_bar = tqdm(range(train_configs.iterations))
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

    # train
    for step in process_bar:
        # mt sampling
        with torch.no_grad():
            preds = model(coords, None)
            sampled_coords, sampled_labels = mt_sampler(coords, labels, preds, train_configs.mt_ratio)

        sampled_preds = model(sampled_coords, None)
        # MSE loss
        loss = ((sampled_preds - sampled_labels) ** 2).mean()

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # process reconstructed image for evaluation
        preds = prep_image_for_eval(preds, model_configs, H, W, C)

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
                        "lr": scheduler.get_last_lr()[0]
                        }
            # Save ground truth image (only at 1st iteration)
            if step == 0:
                save_image_to_wandb(log_dict, ori_img, "GT", dataset_configs, H, W)
                
            # Save reconstructed image
            if step%train_configs.save_interval==0:
                save_image_to_wandb(log_dict, preds, "Reconstruction", dataset_configs, H, W)

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
        best_pred = best_pred.squeeze(-1) if dataset_configs.color_mode == 'L' else best_pred
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

    # Save run name with current time
    time_str = str(datetime.datetime.now().time()).replace(":", "").replace(".", "")
    configs.TRAIN_CONFIGS.out_dir += "_" + time_str

    # Seed python, numpy, pytorch
    seed_everything(configs.TRAIN_CONFIGS.seed)
    # Saving config and settings for reproduction
    save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)

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