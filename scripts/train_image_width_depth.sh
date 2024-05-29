#!/bin/bash

cuda=0
wandb_group=width
wandb_project=network_ablation

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
model="siren"  
depth=("5" "6" "7")  #k
width=("64" "128" "256")  #i

img=("05" "18")
scheduler="constant"   
lr_scheduler="cosine"
strategy=("void" "dense" "incremental")  #j
ratio=0.2   # 1.0 for reverse mode; 0.2 for normal mode

no_io=0

# Train on Kodak dataset
i=2
for img_idx in {0..0}
do 
    for k in {0..2};
    do
        for j in {1..1};
        do
            for top_k in {1..1};
            do
                if [[ top_k -eq 1 || "${strategy[j]}" == "dense"  ]]; then
                    if [[ "${strategy[j]}" == "incremental" ]]; then
                        scheduler="step"
                    else
                        scheduler="constant"
                    fi
                    srun -p utmist --gres gpu python3 train_image.py DATASET_CONFIGS.file_path="../datasets/kodak/kodim${img[img_idx]}.png" \
                    TRAIN_CONFIGS.out_dir="${model}_${width[i]}_${depth[k]}_${strategy[j]}_${scheduler}_${lr_scheduler}_kodak${img[img_idx]}" \
                    TRAIN_CONFIGS.device="cuda:${cuda}" \
                    TRAIN_CONFIGS.sampling_path="logs/sampling/${wandb_group}_${model}_${width[i]}_${depth[k]}_${strategy[j]}_${scheduler}_${lr_scheduler}_kodak${img[img_idx]}.pkl" \
                    TRAIN_CONFIGS.no_io="${no_io}" \
                    EXP_CONFIGS.mt_ratio="${ratio}" \
                    EXP_CONFIGS.scheduler_type="${scheduler}" \
                    EXP_CONFIGS.optimizer_type="adam" \
                    EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
                    EXP_CONFIGS.strategy_type="${strategy[j]}" \
                    EXP_CONFIGS.top_k="${top_k}" \
                    WANDB_CONFIGS.group="${wandb_group}" \
                    WANDB_CONFIGS.wandb_project="${wandb_project}" \
                    +NETWORK_CONFIGS.dim_hidden="${width[i]}" \
                    +NETWORK_CONFIGS.num_layers="${depth[k]}" \
                    model_config="${model}" &> "logs/logs/${wandb_group}_${model}_${width[i]}_${depth[k]}_${strategy[j]}_${scheduler}_${lr_scheduler}_kodak${img[img_idx]}.txt"
                fi
                done
        done
    done
done