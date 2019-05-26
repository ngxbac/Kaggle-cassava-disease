#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
RUN_CONFIG=config.yml

model=resnet50
LOGDIR=/raid/bac/kaggle/logs/cassava-disease/$model
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --model_params/model_name=$model:str \
    --verbose