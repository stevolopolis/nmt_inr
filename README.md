# NMT INR

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

## Updates
6:23pm EST Jan12 
- `Experiment` - General exploration about MT dyanmics and MT wallclock time performance
- `Script` - `scripts/mt_ratio.sh` (x2: different sets of kodak images)
- `Log` - https://wandb.ai/hku_inr/mt_exploration/groups/general