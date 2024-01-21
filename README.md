# NMT INR

## Updates
10:01am EST Jan21
- `Analysis` - Added code to visualize MT progression. RMB to add two subdirectories to run the `analysis/mt_dyanmics.py`:
    - `vis/dynamics`
    - `vis/iou`
6:23pm EST Jan12 
- `Experiment` - General exploration about MT dyanmics and MT wallclock time performance
- `Script` - `scripts/mt_ratio.sh` (x2: different sets of kodak images)
- `Log` - https://wandb.ai/hku_inr/mt_exploration/groups/general

## Code structure
- `train_image.py` - Main training code
- `sampler.py` - Code where we define classes/functions for MT sampling strategies
- `dataset.py` - Dataset loaders (image, video)
- `utils.py` - Miscellaneous functions

## Environment setup
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install conda-forge::einops
conda install conda-forge::easydict
conda install conda-forge::hydra-core
conda install conda-forge::wandb
conda install conda-forge::scikit-learn
conda install anaconda::scikit-image
```
