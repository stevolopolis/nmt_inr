#!/usr/bin/bash

cuda=0
wandb_group=image
wandb_project=nmt

model="siren"
img=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")       # i
lr_scheduler="cosine"
scheduler="step"  
strategy="incremental"
ratio=0.2 
top_k="1"

# Train on Kodak dataset
for i in {0..23};
do
    python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[i]}.png" \
    TRAIN_CONFIGS.out_dir="${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_kodim${img[i]}" \
    TRAIN_CONFIGS.save_name="${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_kodim${img[i]}" \
    TRAIN_CONFIGS.device="cuda:${cuda}" \
    TRAIN_CONFIGS.sampling_path="logs/sampling" \
    TRAIN_CONFIGS.loss_path="logs/loss" \
    EXP_CONFIGS.mt_ratio="${ratio}" \
    EXP_CONFIGS.scheduler_type="${scheduler}" \
    EXP_CONFIGS.optimizer_type="adam" \
    EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
    EXP_CONFIGS.strategy_type="${strategy}" \
    EXP_CONFIGS.top_k="${top_k}" \
    WANDB_CONFIGS.group="${wandb_group}" \
    WANDB_CONFIGS.wandb_project="${wandb_project}" \
    model_config="${model}" &> "logs/logs/${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_kodim${img[i]}.txt"
done