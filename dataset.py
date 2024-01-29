import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchaudio
from PIL import Image
from einops import rearrange
import numpy as np
import skimage
import os


class AudioFileDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(
        self,
        dataset_configs,
        input_output_configs
    ):
        super().__init__(root='data', url='test-clean', download=False)

        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range

        # LibriSpeech contains audio 16kHz rate
        self.sample_rate = 16000
        self.num_secs = dataset_configs.num_secs
        self.num_waveform_samples = int(self.num_secs * self.sample_rate)
        self.sample_idx = dataset_configs.sample_idx

        # __getitem__ returns a tuple, where first entry contains raw waveform in [-1, 1]
        self.labels = super().__getitem__(self.sample_idx)[0].float()

        # Normalize data to lie in [0, 1]
        if self.data_range == 0:
            self.labels = (self.labels + 1) / 2

        # Extract only first num_waveform_samples from waveform
        if self.num_secs != -1:
            # Shape (channels, num_waveform_samples)
            self.labels = self.labels[:, : self.num_waveform_samples].view(-1, 1)

        self.T, self.C = self.num_waveform_samples, 1
        if self.coord_mode == 0:
            grid = [torch.linspace(0.0, self.T-1, self.T)] # [0, T-1]
        elif self.coord_mode == 1:
            grid = [torch.linspace(0., 1., self.T)] # [0, 1]
        elif self.coord_mode == 2:
            grid = [torch.linspace(-5., 5., self.T)] # [-5, 5] following coin++
        elif self.coord_mode == 4:
            grid = [torch.linspace(-0.5, 0.5, self.T)] # [0.5, 0.5]
        else:
            raise NotImplementedError

        self.coords = torch.stack(
            torch.meshgrid(grid),
            dim=-1,
        ).view(-1, 1)

        self.dim_in = 1
        self.dim_out = 1
        

    def __len__(self):
        return 1

    def get_data_shape(self):
        return (self.T, self.C)

    def get_data(self):
        return self.coords, self.labels


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


class MeshSDF(Dataset):
    ''' convert point cloud to SDF '''

    def __init__(self, dataset_configs, input_output_configs):
        super().__init__()
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range

        # self.num_samples = self.config.num_samples
        self.pointcloud_path = self.config.xyz_file
        self.coarse_scale = self.config.coarse_scale
        self.fine_scale = self.config.fine_scale
        self.normalize = True
        self.dim_in = 3
        self.dim_out = 1
        self.out_range = None

        # load gt point cloud with normals
        self.load_mesh(self.config.xyz_file)
        
        # precompute sdf and occupancy grid
        self.render_resolution = self.config.render_resolution
        self.load_precomputed_occu_grid(self.config.xyz_file, self.render_resolution)

    def load_precomputed_occu_grid(self, xyz_file, render_resolution):
        # load from files if exists
        sdf_file = xyz_file.replace('.xyz', f'_{render_resolution}_sdf.npy')
        if os.path.exists(sdf_file):
            self.sdf = np.load(sdf_file)
        else:
            self.sdf = self.build_sdf(render_resolution)
            np.save(sdf_file, self.sdf)

        occu_grid = (self.sdf <= 0)
        self.occu_grid = occu_grid

    def build_sdf(self, render_resolution):
        N = render_resolution
        # build grid
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
        x, y, z = torch.meshgrid(x, x, x)
        render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
        vox_centers = render_coords.cpu().numpy()

        # use KDTree to get nearest neighbours and estimate the normal
        _, idx = self.kd_tree.query(vox_centers, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((vox_centers - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf.reshape(N, N, N)
        return sdf
    
    def build_grid_coords(self, render_resolution):
        N = render_resolution
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
        x, y, z = torch.meshgrid(x, x, x)
        coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
        return coords.cpu().numpy()

    def load_mesh(self, pointcloud_path):
        from pykdtree.kdtree import KDTree
        npy_file = pointcloud_path.replace('.xyz', '.npy')
        if os.path.exists(npy_file):
            pointcloud = np.load(npy_file)
        else:
            pointcloud = np.genfromtxt(pointcloud_path)
            np.save(pointcloud_path.replace('.xyz', '.npy'), pointcloud)
        self.pointcloud = pointcloud
        
        # cache to speed up loading
        self.v = pointcloud[:, :3]
        self.n = pointcloud[:, 3:]

        n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        n_norm[n_norm == 0] = 1.
        self.n = self.n / n_norm
        self.v = self.normalize_coords(self.v)
        self.kd_tree = KDTree(self.v)
        print('finish loading pc')

    def normalize_coords(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def sample_surface(self):
        # idx = np.random.randint(0, self.v.shape[0], self.num_samples)
        points = self.v
        points[::2] += np.random.laplace(scale=self.coarse_scale, size=(points.shape[0] - points.shape[0]//2, points.shape[-1]))
        points[1::2] += np.random.laplace(scale=self.fine_scale, size=(points.shape[0]//2, points.shape[-1]))

        # wrap around any points that are sampled out of bounds
        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.kd_tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]

        return points, sdf

    def __getitem__(self, idx):
        batch_size = 262144
        start_idx = idx * batch_size
        coords, sdf = self.get_data(start_idx, batch_size)
        return coords, sdf
    
    def __len__(self):
        return 1
    
    def get_data(self):
        coords, sdf = self.sample_surface()
        return torch.from_numpy(coords).float(), torch.from_numpy(sdf).float()
