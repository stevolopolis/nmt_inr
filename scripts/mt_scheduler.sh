#!/usr/bin/bash

cuda=0
wandb_group=scheduler

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
img=("05" "18" "20" "21")
model=("mlp" "siren" "ffn")
scheduler=("constant" "linear" "step" "cosine")
lr_scheduler="constant"  # "cosine"

# Train on Kodak dataset
for k in {3..3};
do
    for j in {0..2};
    do 
        for i in {0..0};
        do
            python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[k]}.png" \
            TRAIN_CONFIGS.out_dir="${model[j]}_${scheduler[i]}_${lr_scheduler}_kodak${img[k]}" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            TRAIN_CONFIGS.sampling_path="logs/sampling/scheduler_${model[j]}_${scheduler[i]}_${lr_scheduler}_kodak${img[k]}.pkl" \
            EXP_CONFIGS.scheduler_type="${scheduler[i]}" \
            EXP_CONFIGS.optimizer_type="adam" \
            EXP_CONFIGS.lr_scheduler_type="constant" \
            WANDB_CONFIGS.group="${wandb_group}" \
            model_config="${model[j]}" &> "logs/logs/scheduler_${model[j]}_${scheduler[i]}_${lr_scheduler}_kodak${img[k]}.txt"
        done
    done
done