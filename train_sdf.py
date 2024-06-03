import os
from omegaconf import OmegaConf
from easydict import EasyDict
import wandb
import numpy as np
import torch
import hydra
import logging
from tqdm import tqdm
import yaml
import shutil
from utils import seed_everything, get_dataset, get_model
from sdf_utils.generate_mesh import generate_mesh
from sdf_utils import open3d_utils
import mcubes
import trimesh

from nmt import NMT
from utils import seed_everything, get_dataset, get_model

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

    # prepare training settings
    model.train()
    model = model.to(device)

    process_bar = tqdm(range(train_configs.iterations))
    C = dataset.dim_out
    best_iou, best_cd = 0, float("inf")
    best_mesh = None

    # nmt setup
    nmt = NMT(model,
              train_configs.iterations,
              (C,),
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
    iou_milestone = False

    # train
    for step in process_bar:
        coords, labels = dataset.get_data()
        coords, labels = coords.to(device), labels.to(device)  

        # mt sampling
        sampled_batch_coords, sampled_batch_labels, full_batch_preds = nmt.sample(step, coords, labels)
        
        sampled_batch_preds = model(sampled_batch_coords)

        # MSE loss
        loss = ((sampled_batch_preds - sampled_batch_labels) ** 2).mean()

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if step % train_configs.save_interval == 0 or step == train_configs.iterations-1:
            # compute iou
            pred_mesh, pred_sdf = generate_mesh(model, N=dataset.render_resolution ,return_sdf=True, device=device)
            pred_occ = pred_sdf <= 0
            gt_occ = dataset.occu_grid
            intersection = np.sum(np.logical_and(gt_occ, pred_occ))
            union = np.sum(np.logical_or(gt_occ, pred_occ))
            iou = intersection / union

            # compute chamfer distance
            sdf = dataset.sdf
            vertices, triangles = mcubes.marching_cubes(-sdf, 0)
            N = dataset.render_resolution
            gt_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            gt_mesh.vertices = (gt_mesh.vertices / N - 0.5) + 0.5/N
            print(f"iou: {iou}")

            # W&B logging
            if configs.WANDB_CONFIGS.use_wandb:
                log_dict = {
                            "loss": loss.item(),
                            "iou": iou,
                            "lr": scheduler.get_last_lr()[0],
                            "mt": nmt.get_ratio(),
                            "mt_interval": nmt.get_interval()
                            }
                # Save ground truth image (only at 1st iteration)
                if step == 0:
                    log_dict["GT"] =  wandb.Object3D(gt_mesh.vertices)
                    
                # Save reconstructed 3d shape
                if step%train_configs.save_interval==0:
                    log_dict["Reconstruction"] =  wandb.Object3D(pred_mesh.vertices)

                if not iou_milestone and iou > 0.85:
                    iou_milestone = True
                    wandb.log({"IoU Threshold": step}, step=step)

                wandb.log(log_dict, step=step)

            # Save model weight with best iou
            if iou > best_iou:
                best_iou = iou
                best_mesh = pred_mesh

        # udpate progress bar
        process_bar.set_description(f"loss x 10000: {loss.item()*10000:.4f}, best_iou: {best_iou*100:.2f}")

    # wrap up training
    print("Training finished!")
    print(f"Best iou: {best_iou:.4f}")
    
    # W&B logging of final step 
    if configs.WANDB_CONFIGS.use_wandb:
        wandb.log(
                {
                "best_iou": iou,
                "best_pred": wandb.Object3D(best_mesh.vertices),
                }, 
            step=step)
        wandb.finish()
    log.info(f"Best iou: {best_iou:.4f}")

    # save mesh
    o3d_mesh = open3d_utils.trimesh_to_o3d_mesh(best_mesh)
    os.makedirs(os.path.join('outputs', out_dir, 'meshes'), exist_ok=True)
    open3d_utils.save_mesh(o3d_mesh, os.path.join('outputs', out_dir, 'meshes', 'best_mesh.ply'))

    # save model
    # torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))

    return best_iou, best_cd

@hydra.main(version_base=None, config_path='config', config_name='train_sdf')
def main(configs):
    configs = EasyDict(configs)

    # Seed python, numpy, pytorch
    seed_everything(configs.TRAIN_CONFIGS.seed)
    # Saving config and settings for reproduction
    # save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)

    # model configs
    configs.model_config.INPUT_OUTPUT.data_range = configs.NETWORK_CONFIGS.data_range
    configs.model_config.INPUT_OUTPUT.coord_mode = configs.NETWORK_CONFIGS.coord_mode
    if configs.model_config.name == "FFN":
        configs.model_config.NET.rff_std = configs.NETWORK_CONFIGS.rff_std

    # model and dataloader
    dataset = get_dataset(configs.DATASET_CONFIGS, configs.model_config.INPUT_OUTPUT)
    model = get_model(configs.model_config, dataset)
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")

    n_params = sum([p.numel() for p in model.parameters()])
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
    best_iou, best_cd = train(configs, model, dataset, device=configs.TRAIN_CONFIGS.device)
    log.info(f"No. of parameters: {sum([p.numel() for p in model.parameters()])}")
    log.info(f"Results saved in: {configs.TRAIN_CONFIGS.out_dir}")
    return best_iou, best_cd

if __name__=='__main__':
    main()
