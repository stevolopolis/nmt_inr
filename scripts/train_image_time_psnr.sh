#!/bin/bash

cuda=0
wandb_group=time_psnr
wandb_project=hparam

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
model="siren"  
depth="6"  #k
width="256"  #i

img=("05" "18")
scheduler="constant"   
lr_scheduler="cosine"
strategy=("void" "dense")  #j
ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)  

no_io=0

# Train on Kodak dataset
for j in {1..1};
do
    for top_k in {1..1};
    do
        for i in {4..8};
        do
            for no_io in {0..1};
            do
                if [[ top_k -eq 1 || "${strategy[j]}" == "dense"  ]]; then
                    if [[ "${strategy[j]}" == "incremental" ]]; then
                        scheduler="step"
                    else
                        scheduler="constant"
                    fi
                    mt_ratio="${ratio[i]}"
                    srun -p utmist --gres gpu python3 train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[img_idx]}.png" \
                    TRAIN_CONFIGS.out_dir="${model}_${width}_${depth}_ratio${mt_ratio}_topk${top_k}_noIO${no_io}_kodak${img[img_idx]}" \
                    TRAIN_CONFIGS.device="cuda:${cuda}" \
                    TRAIN_CONFIGS.sampling_path="logs/sampling/${wandb_group}_${model}_${width}_${depth}_ratio${mt_ratio}_topk${top_k}_noIO${no_io}_kodak${img[img_idx]}.pkl" \
                    TRAIN_CONFIGS.no_io="${no_io}" \
                    EXP_CONFIGS.mt_ratio="${mt_ratio}" \
                    EXP_CONFIGS.scheduler_type="${scheduler}" \
                    EXP_CONFIGS.optimizer_type="adam" \
                    EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
                    EXP_CONFIGS.strategy_type="${strategy[j]}" \
                    EXP_CONFIGS.top_k="${top_k}" \
                    WANDB_CONFIGS.group="${wandb_group}" \
                    WANDB_CONFIGS.wandb_project="${wandb_project}" \
                    +NETWORK_CONFIGS.dim_hidden="${width}" \
                    +NETWORK_CONFIGS.num_layers="${depth}" \
                    model_config="${model}" &> "logs/logs/${wandb_group}_${model}_${width}_${depth}_ratio${mt_ratio}_topk${top_k}_noIO${no_io}_kodak${img[img_idx]}.txt"
                fi
            done
        done
    done
done