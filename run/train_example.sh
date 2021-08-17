#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8
MOBILE_MEMORY=202108

OUTPUT=./CLS/ckpt/checkpoint_${MOBILE_MEMORY}/
mkdir -p ${OUTPUT}

LOG=./CLS/logs/logs_${MOBILE_MEMORY}
mkdir -p ${LOG}

python3 -u -m torch.distributed.launch --nproc_per_node=8 --master_port 11111 train.py \
    --img-root-dir  /home/data4/OULU \
    --train-file-path  ./data/list/p1_train_list.txt \
    --evaluate \
    --eval-interval-time 1 \
    --val-file-path  ./data/list/p1_dev_list.txt \
    --arch deit \
    --input-size  224 \
    --crop-scale 1.5 \
    --pth-save-dir  ${OUTPUT} \
    --log-save-dir ${LOG} \
    --pth-save-iter  1000 \
    --epochs  100 \
    --batch-size 32 \
    --learning-rate  0.00067 \
    --weight-decay 1e-4 \
    --optimizer-type  adamw \
    --loss-type  lsce \
    --lr-schedule cosine \
    --workers 32  \
