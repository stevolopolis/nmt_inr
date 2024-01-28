#!/usr/bin/bash

cuda=1
wandb_group=kodak
wandb_project=strategic_mt

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
model=("siren" "ffn")     # t
#img=("05" "18" "20" "21" "07" "14" "15" "17")      # k
img=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")
scheduler="step"   
lr_scheduler="cosine"
strategy=("void" "incremental" "dense")    # j
ratio=0.2   # 1.0 for reverse mode; 0.2 for normal mode

# Train on Kodak dataset
for t in {0..0};
do
    for k in {0..0};
    do
        for j in {1..1};
        do
            python train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[k]}.png" \
            TRAIN_CONFIGS.out_dir="${model[t]}_${strategy[j]}_${scheduler}_${lr_scheduler}_kodak${img[k]}" \
            TRAIN_CONFIGS.device="cuda:${cuda}" \
            TRAIN_CONFIGS.sampling_path="logs/sampling/${wandb_group}_${model[t]}_${strategy[j]}_${scheduler}_${lr_scheduler}_kodak${img[k]}.pkl" \
            EXP_CONFIGS.mt_ratio="${ratio}" \
            EXP_CONFIGS.scheduler_type="${scheduler}" \
            EXP_CONFIGS.optimizer_type="adam" \
            EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
            EXP_CONFIGS.strategy_type="${strategy[j]}" \
            WANDB_CONFIGS.group="${wandb_group}" \
            WANDB_CONFIGS.wandb_project="${wandb_project}" \
            model_config="${model[t]}" &> "logs/logs/${wandb_group}_${model[t]}_${strategy[j]}_${scheduler}_${lr_scheduler}_kodak${img[k]}.txt"
        done
    done    
done