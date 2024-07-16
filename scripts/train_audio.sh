#!/usr/bin/bash

cuda=0
wandb_group=audio
wandb_project=nmt

model="siren"
lr_scheduler="cosine"
scheduler="step"   
strategy="incremental"
ratio=0.2   
top_k="1"   

# Train on Kodak dataset
for k in {0..99};
do
    python train_audio.py DATASET_CONFIGS.sample_idx=${k} \
    TRAIN_CONFIGS.out_dir="audio_${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_idx${k}" \
    TRAIN_CONFIGS.save_name="audio_${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_idx${k}" \
    TRAIN_CONFIGS.device="cuda:${cuda}" \
    TRAIN_CONFIGS.sampling_path="logs/sampling" \
    TRAIN_CONFIGS.loss_path="logs/loss" \
    EXP_CONFIGS.mt_ratio="${ratio}" \
    EXP_CONFIGS.scheduler_type="${scheduler}" \
    EXP_CONFIGS.optimizer_type="adam" \
    EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
    EXP_CONFIGS.strategy_type="${strategy}" \
    WANDB_CONFIGS.group="${wandb_group}" \
    WANDB_CONFIGS.wandb_project="${wandb_project}"
done