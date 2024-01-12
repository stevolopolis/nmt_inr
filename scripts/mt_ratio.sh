#!/usr/bin/bash

cuda=1
wandb_group=general

model=("mlp" "siren" "ffn")
img=("07" "14" "15" "17")  # ("05" "18" "20" "21") [cuda:0]
ratio=("1.0" "0.2" "0.4" "0.6" "0.8")

# Train on Kodak dataset
for k in {0..3};
do
    for j in {0..3};
    do 
        for i in {0..5};
        do
            python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[k]}.png" \
            TRAIN_CONFIGS.out_dir="${model[j]}_mt${ratio[i]}_kodak${img[k]}" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            TRAIN_CONFIGS.mt_ratio="${ratio[i]}" \
            WANDB_CONFIGS.group="${wandb_group}" \
            model_config="${model[j]}" &> "logs/log_${wandb_group}_${model[j]}_${img[k]}.txt "
        done
    done
done