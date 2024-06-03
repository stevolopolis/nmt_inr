#!/usr/bin/bash
cuda=0
wandb_group=audio

model=("siren" "ffn")       # j
ratio=("1.0" "0.2" "0.4" "0.6" "0.8")       # i
lr=("1e-4" "1e-3")      # j

scheduler="step"
lr_scheduler="cosine"
strategy="incremental"
topk="1"

# Train on first 100 samples (2s) from the test-clean split of LibriSpeech dataset
for k in {0..0};
do
    for j in {0..0};
    do 
        for i in {1..1};
        do
            python train_audio.py DATASET_CONFIGS.sample_idx=${k} \
            TRAIN_CONFIGS.out_dir="${model}_mt${ratio[i]}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_libri${k}" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            TRAIN_CONFIGS.sampling_path="logs/sampling" \
            TRAIN_CONFIGS.loss_path="logs/loss" \
            TRAIN_CONFIGS.save_name="${model}_mt${ratio[i]}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_libri${k}" \
            NETWORK_CONFIGS.lr="${lr[j]}" \
            EXP_CONFIGS.mt_ratio="${ratio[i]}" \
            EXP_CONFIGS.scheduler_type="${scheduler}" \
            EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
            EXP_CONFIGS.strategy_type="${strategy}" \
            WANDB_CONFIGS.group="${wandb_group}" \
            model_config="${model[j]}" &> "logs/logs/log_${model}_mt${ratio[i]}_${strategy}_${scheduler}_${lr_scheduler}_topk${top_k}_libri${k}.txt"
        done
    done
done