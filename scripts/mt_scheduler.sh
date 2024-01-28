#!/usr/bin/bash

cuda=0
wandb_group=sampling_vis

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
ratio=('0.2' '0.4' '0.6' '0.8')     # t
img=("05" "18" "20" "21" "07" "14" "15" "17")      # k
model=("siren" "ffn")     # j
scheduler=("constant" "linear" "step" "cosine")     # i
lr_scheduler="constant"  # "cosine"

# Train on Kodak dataset
for t in {0..1};
do
    for k in {0..0};
    do
        for j in {0..0};
        do 
            for i in {0..0};
            do
                python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[k]}.png" \
                TRAIN_CONFIGS.out_dir="sampling_${model[j]}_mt${ratio[t]}_${scheduler[i]}_${lr_scheduler}_kodak${img[k]}" \
                TRAIN_CONFIGS.device="cuda:${cuda}" \
                TRAIN_CONFIGS.sampling_path="logs/sampling/sampling_${model[j]}_mt${ratio[t]}_${scheduler[i]}_${lr_scheduler}_kodak${img[k]}_2.pkl" \
                TRAIN_CONFIGS.loss_path="logs/loss/loss_${model[j]}_mt${ratio[t]}_${scheduler[i]}_${lr_scheduler}_kodak${img[k]}_2.pkl" \
                EXP_CONFIGS.mt_ratio="${ratio[t]}" \
                EXP_CONFIGS.scheduler_type="${scheduler[i]}" \
                EXP_CONFIGS.optimizer_type="adam" \
                EXP_CONFIGS.lr_scheduler_type="constant" \
                WANDB_CONFIGS.group="${wandb_group}" \
                model_config="${model[j]}" &> "logs/logs/sampling_${model[j]}_mt${ratio[t]}_${scheduler[i]}_${lr_scheduler}_kodak${img[k]}_2.txt"
            done
        done
    done
done