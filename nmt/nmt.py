import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple
from pathlib import Path

from .scheduler import *
from .sampler import mt_sampler, save_samples, save_losses
from .strategy import strategy_factory

class NMT:
    """
    Wrapper class for the NMT algorithm.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 iters: int,
                 data_shape: Union[Tuple[int], List[int]],
                 scheduler: str="step",
                 strategy: str="incremental",
                 starting_ratio: float=0.2,
                 top_k: bool=True,
                 save_samples_path: Path=Path("logs/sampling"),
                 save_losses_path: Path=Path("logs/losses"),
                 save_name: str=None,
                 save_interval: int=1000):
        self.model = model
        self.scheduler = mt_scheduler_factory(scheduler)
        self.strategy = strategy_factory(strategy)
        
        self.iters = iters
        self.starting_ratio = starting_ratio
        self.ratio = starting_ratio
        self.mt_interval = None
        self.top_k = top_k
        if len(data_shape) == 3:
            self.H, self.W, self.C = data_shape
        else:
            self.H = None
            self.W = None
            self.C = None

        self.preds = None   
        self.sampled_x = None
        self.sampled_y = None

        self.save_interval = save_interval
        if save_samples_path is not None:
            self.sampling_path = Path(save_samples_path) if type(save_samples_path) is not Path else save_samples_path
            self.sampling_path.mkdir(parents=True, exist_ok=True)
        else:
            self.sampling_path = None
        if save_losses_path is not None:
            self.loss_path = Path(save_losses_path) if type(save_losses_path) is not Path else save_losses_path
            self.loss_path.mkdir(parents=True, exist_ok=True)
        else:
            self.loss_path = None
        self.save_name = f"mt${starting_ratio}_${strategy}_${scheduler}_topk${top_k}" if save_name is None else save_name

        self.save_sample_path = None
        self.save_loss_path = None
        self.save_tint_path = None

        self.sampling_history = dict()
        self.loss_history = dict()

    def sample(self, iter: int, x: torch.tensor, y: torch.tensor):
        self.ratio = self.scheduler(iter, self.iters, self.starting_ratio)
        mt, self.mt_intervals = self.strategy(iter, self.iters)
        if mt:
            with torch.no_grad():
                self.preds = self.model(x)
                self.sampled_x, self.sampled_y, idx, dif = mt_sampler(x, y, self.preds, self.ratio, self.top_k)
                if iter % self.save_interval == 0:
                    if self.sampling_path is not None:
                        self.save_sample_path = self.sampling_path / f"{self.save_name}_samples.pkl"
                        self.save_tint_path = self.sampling_path / f"{self.save_name}_tint_{iter}.png"

                        save_samples(self.sampling_history, iter, self.iters, self.sampled_x, self.save_sample_path)

                        if self.H is not None and self.W is not None and self.C is not None:
                            tinted_x = self._tint_data_with_samples(y, idx)
                            tinted_img = self._preprocess_img(tinted_x, self.H, self.W, self.C)
                            self._save_image(tinted_img, self.save_tint_path, self.H, self.W, color_mode="L" if self.C == 1 else "RGB")
                    if self.loss_path is not None:
                        self.save_loss_path = self.loss_path / f"{self.save_name}_losses.pkl"
                        save_losses(self.loss_history, iter, self.iters, dif, self.save_loss_path)

        elif not mt and self.mt_intervals is None:
            self.sampled_x = x
            self.sampled_y = y
            self.preds = y

        return self.sampled_x, self.sampled_y, self.preds

    def get_ratio(self):
        return self.ratio
    
    def get_interval(self):
        return self.mt_intervals

    def get_saved_samples_path(self):
        return self.save_sample_path

    def get_saved_losses_path(self):
        return self.save_loss_path

    def get_saved_tint_path(self):
        return self.save_tint_path

    def _tint_data_with_samples(self, data, sample_idx, tint_color: List[float]=[0.5, 0.0, 0.0]):
        """Relabel the data with given vis_label at the sample_idx indices."""
        if sample_idx is None: 
            return None
        
        new_data = data.detach().clone()
        vis_label = torch.tensor(tint_color).to(data.device)
        if data.shape[-1] == 1:
            vis_label = vis_label[0]

        new_data[sample_idx] = torch.clamp(new_data[sample_idx] + vis_label, max=1.0)

        return new_data
    
    def _preprocess_img(self, image, h, w, c):
        if torch.min(image) < 0:
            image = image.clamp(-1, 1).view(h, w, c)       # clip to [-1, 1]
            image = (image + 1) / 2                        # [-1, 1] -> [0, 1]
        else:
            image = image.clamp(0, 1).view(h, w, c)       # clip to [0, 1]

        image = image.cpu().detach().numpy()

        return image
    
    def _save_image(self, img, path, h, w, color_mode="RGB"):
        img = Image.fromarray((img *255).astype(np.uint8), mode=color_mode)
        if img.size[0] > 512:
            img = img.resize((512, int(512*h/w)), Image.LANCZOS)
        img.save(path)
        print(f"Image saved to {path}")