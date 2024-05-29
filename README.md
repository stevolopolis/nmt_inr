<div align="center">
  <img src="https://github.com/stevolopolis/nmt_inr/blob/main/asset/int_logo.png" width="10%">
</div>



# NMT INR

[Chen Zhang](https://chen2hang.github.io/), [S.T.S Luo](https://www.cs.toronto.edu/~stevenlts/index.html), [Jason Chun Lok Li](https://hk.linkedin.com/in/jason-chun-lok-li-0590b3166), [Yik-Chung Wu](https://www.eee.hku.hk/~ycwu/), [Ngai Wong](https://www.eee.hku.hk/~nwong/)

[[`Paper`](https://arxiv.org/pdf/2405.10531)]

PyTorch implementation the INT algorithm for INRs. For details, see the paper **[Nonparametric Teaching of Implicit Neural Representations](https://arxiv.org/pdf/2405.10531)**.


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
