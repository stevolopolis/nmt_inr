#!/usr/bin/bash

cuda=0
wandb_group=megapixel
wandb_project=nmt

model="siren"
lr_scheduler="cosine"
scheduler="step"
strategy="incremental"
ratio="0.2"
top_k="1"    

iterations=500

num_layers=6
dim_hidden=512
max_coords=524288


# Train on megapixel image
python train_megapixel.py \
TRAIN_CONFIGS.out_dir="pluto_${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}" \
TRAIN_CONFIGS.save_name="pluto_${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}" \
TRAIN_CONFIGS.device="cuda:${cuda}" \
TRAIN_CONFIGS.sampling_path="logs/sampling" \
TRAIN_CONFIGS.loss_path="logs/loss" \
TRAIN_CONFIGS.iterations="${iterations}" \
EXP_CONFIGS.mt_ratio="${ratio}" \
EXP_CONFIGS.scheduler_type="${scheduler}" \
EXP_CONFIGS.optimizer_type="adam" \
EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
EXP_CONFIGS.strategy_type="${strategy}" \
EXP_CONFIGS.top_k="${top_k}" \
NETWORK_CONFIGS.num_layers="${num_layers}" \
NETWORK_CONFIGS.dim_hidden="${dim_hidden}" \
DATASET_CONFIGS.max_coords="${max_coords}" \
WANDB_CONFIGS.group="${wandb_group}" \
WANDB_CONFIGS.wandb_project="${wandb_project}" \
model_config="${model}" &> "logs/logs/pluto_${model}_mt${ratio}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}.txt"
