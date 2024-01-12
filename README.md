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

## Current train_image.py and bash script
Experiment - basic visualization of MT dynamics at different sampling ratios {0.2, 0.4, 0.6, 0.8, 1.0}.
Results at https://wandb.ai/hku_inr/mt_exploration/groups/ratio