#!/usr/bin/bash
cuda=0
wandb_group=audio

model=("siren" "ffn")
ratio=("1.0" "0.2" "0.4" "0.6" "0.8")
lr=("1e-4" "1e-3")

# Train on first 100 samples (2s) from the test-clean split of LibriSpeech dataset
for k in {0..99};
do
    for j in {0..2};
    do 
        for i in {0..5};
        do
            python train_audio.py DATASET_CONFIGS.sample_idx=${k} \
            TRAIN_CONFIGS.out_dir="${model[j]}_mt${ratio[i]}_libri${k}" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            TRAIN_CONFIGS.lr="${lr[j]}" \
            TRAIN_CONFIGS.mt_ratio="${ratio[i]}" \
            WANDB_CONFIGS.group="${wandb_group}" \
            model_config="${model[j]}" &> "logs/log_${wandb_group}_${model[j]}_${k}.txt "
        done
    done
done