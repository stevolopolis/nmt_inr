#!/usr/bin/bash

cuda=1
model=siren
wandb_group=ratio

img=01
ratio=("1.0" "0.2" "0.4" "0.6" "0.8")

# Train on Kodak dataset
for i in {0..5};
do
    python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img}.png" \
    TRAIN_CONFIGS.out_dir="${model}_mt${ratio[i]}_kodak$i" \
    TRAIN_CONFIGS.device="cuda:${cuda}" \
    TRAIN_CONFIGS.mt_ratio="${ratio[i]}" \
    WANDB_CONFIGS.group="${wandb_group}" \
    model_config="${model}" &> log.txt 
done