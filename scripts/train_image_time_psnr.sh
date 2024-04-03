#!/bin/bash

cuda=2
wandb_group=time_psnr
wandb_project=hparam

# full training set
# ("07" "14" "15" "17")
# ("05" "18" "20" "21")
model="siren"  
depth="6"
width="256"

img=("01" "02" "03" "04" "06" "08" "09" "10" "11")
# img=("21" "20" "13" "22" "23" "24" "16" "19" "12")
scheduler=("constant" "linear")
lr_scheduler="cosine"
strategy=("void" "dense" "dense2")  #j
ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)  #i

no_io=0
scheduler_id=0

# Train on Kodak dataset
for img_idx in {0..8};
do
    for j in {1..1};
    do
        for i in {0..9};
        do
            for top_k in {0..1};
            do
                for no_io in {0..0};
                do
                    if [[ "${ratio[i]}" == "1.0" ]]; then
                        j=0
                    fi
                    if [[ top_k -eq 1 || "${strategy[j]}" == "dense"  ]]; then
                        if [[ "${strategy[j]}" == "incremental" ]]; then
                            scheduler="step"
                        else
                            scheduler="constant"
                        fi
                        mt_ratio="${ratio[i]}"
                        if [[ "${strategy[j]}" == "void" ]]; then
                            name="${model}_${width}_${depth}_void_noIO${no_io}_kodak${img[img_idx]}"
                        elif [[ "${scheduler[scheduler_id]}" == "linear" ]]; then
                            name="${model}_${width}_${depth}_ratio${mt_ratio}_topk${top_k}_noIO${no_io}_linearMT_kodak${img[img_idx]}"
                        else
                            name="${model}_${width}_${depth}_ratio${mt_ratio}_topk${top_k}_noIO${no_io}_kodak${img[img_idx]}"
                        fi
                        python train_image.py DATASET_CONFIGS.file_path="../../datasets/kodak/kodim${img[img_idx]}.png" \
                        TRAIN_CONFIGS.out_dir="${name}" \
                        TRAIN_CONFIGS.device="cuda:${cuda}" \
                        TRAIN_CONFIGS.sampling_path="logs/sampling/${wandb_group}_${name}.pkl" \
                        TRAIN_CONFIGS.no_io="${no_io}" \
                        EXP_CONFIGS.mt_ratio="${mt_ratio}" \
                        EXP_CONFIGS.scheduler_type="${scheduler[scheduler_id]}" \
                        EXP_CONFIGS.optimizer_type="adam" \
                        EXP_CONFIGS.lr_scheduler_type="${lr_scheduler}" \
                        EXP_CONFIGS.strategy_type="${strategy[j]}" \
                        EXP_CONFIGS.top_k="${top_k}" \
                        WANDB_CONFIGS.group="${wandb_group}" \
                        WANDB_CONFIGS.wandb_project="${wandb_project}" \
                        +NETWORK_CONFIGS.dim_hidden="${width}" \
                        +NETWORK_CONFIGS.num_layers="${depth}" \
                        model_config="${model}" &> "logs/logs/${wandb_group}_${name}.txt"
                    fi
                done
            done
        done
    done
done