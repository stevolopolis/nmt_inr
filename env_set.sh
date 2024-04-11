# Create a new conda environment
conda create -n int python=3.9

# Activate the environment
conda activate int

# Install PyTorch and related packages
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install additional packages from conda-forge
conda install conda-forge::einops
conda install conda-forge::easydict
conda install conda-forge::hydra-core
conda install conda-forge::wandb
conda install conda-forge::scikit-learn

# Install scikit-image from the anaconda channel
conda install anaconda::scikit-image
