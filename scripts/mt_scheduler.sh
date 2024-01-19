#!/usr/bin/bash

cuda=1
wandb_group=scheduler

model=("mlp" "siren" "ffn")
# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
img=("05" "18" "20" "21") # [cuda:0]
scheduler=("constant" "linear" "step" "cosine")

# Train on Kodak dataset
for k in {0..3};
do
    for j in {0..3};
    do 
        for i in {0..4};
        do
            python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[k]}.png" \
            TRAIN_CONFIGS.out_dir="${model[j]}_${scheduler[i]}_kodak${img[k]}" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            EXP_CONFIGS.scheduler_type="${scheduler[i]}" \
            WANDB_CONFIGS.group="${wandb_group}" \
            model_config="${model[j]}" &> "logs/log_${model[j]}_${scheduler[i]}_kodak${img[k]}.txt "
        done
    done
done