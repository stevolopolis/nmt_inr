import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from einops import rearrange
import numpy as np
import skimage
import skimage


class ImageFileDataset(Dataset):
    def __init__(self, dataset_configs, input_output_configs):
        super().__init__()
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range
        self.color_mode = dataset_configs.color_mode
        
        if 'camera' in dataset_configs.file_path:
            img = Image.fromarray(skimage.data.camera())
            assert dataset_configs.color_mode == 'L', "camera dataset is in grayscale"
        else:
            img = Image.open(dataset_configs.file_path)
            img = img.convert(self.color_mode)
        
        print(dataset_configs.img_size)
        if dataset_configs.img_size is not None:
            img = img.resize(dataset_configs.img_size)

        self.img = img
        self.img_size = img.size
        print(self.img_size)

        img_tensor = ToTensor()(img) # [0, 1]

        if self.data_range == 1:
            img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        img_tensor = rearrange(img_tensor, 'c h w -> (h w) c')
        self.labels = img_tensor

        W, H = self.img_size
        if self.coord_mode == 0:
            grid = [torch.linspace(0.0, H-1, H), torch.linspace(0.0, W-1, W)] # [0, H-1] x [0, W-1]
        elif self.coord_mode == 1:
            grid = [torch.linspace(0., 1., H), torch.linspace(0., 1., W)] # [0, 1]^2
        elif self.coord_mode == 2:
            grid = [torch.linspace(-1., 1., H), torch.linspace(-1., 1., W)] # [-1, 1]^2
        elif self.coord_mode == 3:
            grid = [torch.linspace(-1., 1. - 1e-6, H), torch.linspace(-1., 1. - 1e-6, W)] # [-1, 0.999999]^2
        elif self.coord_mode == 4:
            grid = [torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W)] # [0.5, 0.5]^2

        self.coords = torch.stack(
            torch.meshgrid(grid),
            dim=-1,
        ).view(-1, 2)

        self.H, self.W = H, W
        self.dim_in = 2
        self.dim_out = 3 if self.color_mode == 'RGB' else 1
        self.C = self.dim_out

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_h(self):
        return self.H
    
    def get_w(self):
        return self.W
    
    def get_c(self):
        return self.C

    def get_data_shape(self):
        return (self.H, self.W, self.C)

    def get_data(self):
        return self.coords, self.labels


class VideoFileDataset(Dataset):
    def __init__(self, dataset_configs, input_output_configs):
        super().__init__()
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range

        video_tensor = torch.tensor(np.load(dataset_configs.file_path)) # video_file as numpy array [T, H, W, C]
        self.T, self.H, self.W, self.C = video_tensor.shape
        self.video = video_tensor

        if self.data_range == 2:
            self.video = video_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        mesh_sizes = self.video.shape[:-1]
        if self.coord_mode == 0:
            grid = [torch.linspace(0.0, s-1, s) for s in mesh_sizes] # [0, T-1] x [0, H-1] x [0, W-1]
        elif self.coord_mode == 1:
            grid = [torch.linspace(0., 1., s) for s in mesh_sizes] # [0, 1]^3
        elif self.coord_mode == 3:
            grid = [torch.linspace(-1., 1. - 1e-4, s) for s in mesh_sizes] # [-1, 0.999999]^2
        elif self.coord_mode == 4:
            grid = [torch.linspace(-0.5, 0.5, s) for s in mesh_sizes] # [0.5, 0.5]^2
        else:
            grid = [torch.linspace(-1., 1., s) for s in mesh_sizes] # [-1, 1]^3

        self.coords = torch.stack(
            torch.meshgrid(grid),
            dim=-1
        ).view(-1, 3)

        self.labels = self.video.view(-1, self.C)

        self.dim_in = 3
        self.dim_out = 3

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return self.coords[idx], self.labels[idx]

    def get_t(self):
        return self.T

    def get_h(self):
        return self.H
    
    def get_w(self):
        return self.W
    
    def get_c(self):
        return self.C

    def get_data_shape(self):
        return (self.T, self.H, self.W, self.C)

    def get_data(self):
        return self.coords, self.labels


def save_img(img, filename):
    ''' given np array, convert to image and save '''
    img = Image.fromarray((255*img).astype(np.uint8))
    img.save(filename)

