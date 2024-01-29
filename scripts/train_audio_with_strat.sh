#!/usr/bin/bash
cuda=2
wandb_group=siren_mt
wandb_project=audio

model=("siren")     # t
scheduler="step"   
lr_scheduler="cosine"
strategy=("void" "incremental" "dense")    # j
ratio=0.2   # 1.0 for reverse mode; 0.2 for normal mode

# Train on Kodak dataset
for t in {0..0};
do
    for k in {0..100};
    do
        for j in {1..1};
        do
            python train_audio.py DATASET_CONFIGS.sample_idx=${k} \
            TRAIN_CONFIGS.out_dir="${model[t]}_${strategy[j]}_${scheduler}_${lr_scheduler}_idx${k}" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            TRAIN_CONFIGS.sampling_path="logs/sampling/${wandb_group}_${model[t]}_${strategy[j]}_${scheduler}_${lr_scheduler}_idx${k}.pkl" \
            EXP_CONFIGS.mt_ratio="${ratio}" \
            EXP_CONFIGS.scheduler_type="${scheduler}" \
            EXP_CONFIGS.optimizer_type="adam" \
            EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
            EXP_CONFIGS.strategy_type="${strategy[j]}" \
            WANDB_CONFIGS.group="${wandb_group}" \
            WANDB_CONFIGS.wandb_project="${wandb_project}"
        done
    done    
done