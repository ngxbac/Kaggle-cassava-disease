#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
RUN_CONFIG=config.yml

for model in resnet50 se_resnet50 densenet121; do
    for fold in 0 1 2 3 4; do
        LOGDIR=/raid/bac/kaggle/logs/cassava-disease/finetune/$model/fold_$fold/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --model_params/model_name=$model:str \
            --stages/data_params/train_csv=/raid/bac/kaggle/cassava-disease/notebooks/csv/train_$fold.csv:str \
            --stages/data_params/valid_csv=/raid/bac/kaggle/cassava-disease/notebooks/csv/valid_$fold.csv:str \
            --verbose
    done
done