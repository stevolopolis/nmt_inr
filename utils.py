import random 
import wandb
import numpy as np
import torch
from dataset import *


def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed) # for random partitioning
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prep_audio_for_eval(audio, config, t, c):
    if config.INPUT_OUTPUT.data_range == 1:
        audio = audio.clamp(-1, 1).view(t, c)       # clip to [-1, 1]
    else: # data range == 0
        audio = audio.clamp(0, 1).view(t, c)       # clip to [0, 1]
        audio = audio*2 - 1                         # [0, 1] -> [-1, 1]

    audio = audio.flatten().cpu().detach().numpy()
    return audio

def prep_image_for_eval(image, config, h, w, c):
    if config.INPUT_OUTPUT.data_range == 1:
        image = image.clamp(-1, 1).view(h, w, c)       # clip to [-1, 1]
        image = (image + 1) / 2                        # [-1, 1] -> [0, 1]
    else:
        image = image.clamp(0, 1).view(h, w, c)       # clip to [0, 1]

    image = image.cpu().detach().numpy()

    return image


def save_image_to_wandb(wandb_dict, image, label, dataset_configs, h, w):
    wandb_img = Image.fromarray((image *255).astype(np.uint8), mode=dataset_configs.color_mode)
    if wandb_img.size[0] > 512:
        wandb_img = wandb_img.resize((512, int(512*h/w)), Image.LANCZOS)
    wandb_dict[label] = wandb.Image(wandb_img)


def get_dataset(dataset_configs, input_output_configs):
    if dataset_configs.data_type == "video":
        dataset = VideoFileDatasetAlt(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "image":
        dataset = ImageFileDataset(dataset_configs, input_output_configs)
    elif dataset_configs.data_type == "audio":
        dataset = AudioFileDataset(dataset_configs, input_output_configs) 
    else:
         raise NotImplementedError(f"Dataset {dataset_configs.data_type} not implemented")
    return dataset


def get_model(model_configs, dataset):
    if model_configs.name == 'SIREN':
        from models.siren import Siren
        model = Siren(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            siren_configs=model_configs
        )
    elif model_configs.name == 'FFN':
        from models.ffn import FFN
        model = FFN(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            ffn_configs=model_configs
        )
    elif model_configs.name == "WIRE":
        from models.wire import Wire
        model = Wire(
           in_features=dataset.dim_in, 
           out_features=dataset.dim_out,
           wire_configs=model_configs
        )
    elif model_configs.name == "MLP":
        from models.mlp import MLP
        model = MLP(
           dim_in=dataset.dim_in, 
           dim_out=dataset.dim_out,
           mlp_configs=model_configs
        )
    else:
        raise NotImplementedError(f"Model {model_configs.name} not implemented")
            
    return model