#!/usr/bin/bash

cuda=2
wandb_group=ratio_sgd

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
img=("05" "18" "20" "21")
model=("mlp" "siren" "ffn")
ratio=("1.0" "0.2" "0.4" "0.6" "0.8")

# Train on Kodak dataset
for k in {0..3};
do
    for j in {0..2};
    do 
        for i in {0..4};
        do
            python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[k]}.png" \
            TRAIN_CONFIGS.out_dir="${model[j]}_mt${ratio[i]}_kodak${img[k]}" \
            TRAIN_CONFIGS.sampling_path="logs/sampling/ratio_${model[j]}_${ratio[i]}_kodak${img[k]}.pkl" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            EXP_CONFIGS.mt_ratio="${ratio[i]}" \
            EXP_CONFIGS.optimizer_type="sgd" \
            WANDB_CONFIGS.group="${wandb_group}" \
            model_config="${model[j]}" &> "logs/logs/ratio_${wandb_group}_${model[j]}_${ratio[i]}_${img[k]}.txt "
        done
    done
done