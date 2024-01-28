#!/usr/bin/bash

cuda=0

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
img='05'
model=("siren" "ffn")     # t
lr=('0.01' '0.001' '0.0001' '0.00001')     # k
coord_mode="2"     
data_range="0"     
rff_std=("10" "11" "12" "13" "14")  # i
lr_scheduler="constant"  # "cosine"
scheduler="constant"
strategy="void"
ratio=("0.2" "0.4" "0.6" "0.8") # j
iterations=2500

wandb_project=sgd

#TRAIN_CONFIGS.out_dir="${model[t]}_${rff_std[i]}_lr${lr[k]}_${coord_mode[j]}_kodak${img}" \
#NETWORK_CONFIGS.rff_std="${rff_std[i]}" \
# Train on Kodak dataset
for t in {0..0};
do 
    for k in {0..3};
    do
        for j in {0..3};
        do
            python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img}.png" \
            TRAIN_CONFIGS.out_dir="${model[t]}_mt${ratio[j]}_lr${lr[k]}_camera" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            TRAIN_CONFIGS.sampling_path="logs/sampling/sweep_${model[t]}_mt${ratio[j]}_lr${lr[k]}_camera.pkl" \
            TRAIN_CONFIGS.loss_path="logs/loss/sweep_${model[t]}_mt${ratio[j]}_lr${lr[k]}_camera.pkl" \
            TRAIN_CONFIGS.iterations="${iterations}" \
            EXP_CONFIGS.mt_ratio="${ratio}" \
            EXP_CONFIGS.scheduler_type="${scheduler}" \
            EXP_CONFIGS.optimizer_type="sgd" \
            EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
            EXP_CONFIGS.strategy_type="${strategy}" \
            NETWORK_CONFIGS.lr="${lr[k]}" \
            NETWORK_CONFIGS.data_range="${data_range}" \
            NETWORK_CONFIGS.coord_mode="${coord_mode}" \
            WANDB_CONFIGS.group="${model[t]}" \
            WANDB_CONFIGS.wandb_project="${wandb_project}" \
            model_config="${model[t]}" &> "logs/logs/sweep_${model[t]}_mt${ratio[j]}_lr${lr[k]}_camera.pkl"
        done
    done
done