<div align="center">
  <img src="https://github.com/stevolopolis/nmt_inr/blob/main/asset/int_logo.png" width="15%">
</div>



# NMT INR

[Chen Zhang](https://chen2hang.github.io/), [S.T.S Luo](https://www.cs.toronto.edu/~stevenlts/index.html), [Jason Chun Lok Li](https://hk.linkedin.com/in/jason-chun-lok-li-0590b3166), [Yik-Chung Wu](https://www.eee.hku.hk/~ycwu/), [Ngai Wong](https://www.eee.hku.hk/~nwong/)

[[`Paper`](https://arxiv.org/pdf/2405.10531)]

PyTorch implementation the INT algorithm for INRs. For details, see the paper **[Nonparametric Teaching of Implicit Neural Representations](https://arxiv.org/pdf/2405.10531)**.


## Environment setup
Install dependencies
```
conda env create -f environment.yml
conda activate NMT
```
Install the NMT package
```
python setup.py install
```


## Using the NMT Sampling algorithm
Please view the README in `src/` and read the docstrings of the NMT class in `src/nmt.py`.

## Reproducing published results
The `train_<data>.py` files are training codes for the specific data modalities reported in the paper, while the bash scripts in `scripts/` are used to produce the results. The configuration files (i.e. model config, experimental config, etc.) are located in `config` in the form of `.yaml` files. The config files in this repo do not contain the configurations for all the reported experiments. To reproduce the experimental results, please refer to the paper and amend the configuration files accordingly. 

## Coming soon
Optimized NMT algo
- if mt ratio > ??, do only 1 forward pass (with gradients) and use subset of gradients for backward pass
- if mt ratio < ??, do 1 forward pass (without gradients) and do another subset forward pass (with gradients) for backward pass