#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8
MOBILE_MEMORY=20210708

OUTPUT=/home/projects/face_liveness/FAS/ckpt/checkpoint_${MOBILE_MEMORY}/
mkdir -p ${OUTPUT}

LOG=/home/projects/face_liveness/FAS/logs/logs_${MOBILE_MEMORY}
mkdir -p ${LOG}

python3 -u -m torch.distributed.launch --nproc_per_node=8 --master_port 11111 train.py \
    --img-root-dir  /home/data4/OULU \
    --train-file-path  ./data/list_oulu/p1_train_list.txt \
    --evaluate \
    --eval-interval-time 1 \
    --val-file-path  ./data/list_oulu/p1_dev_list.txt \
    --arch swin \
    --pretrained \
    --input-size  224 \
    --crop-scale 1.3 \
    --pth-save-dir  ${OUTPUT} \
    --log-save-dir ${LOG} \
    --pth-save-iter  1000 \
    --epochs  150 \
    --batch-size 16 \
    --learning-rate  0.0002 \
    --weight-decay 1e-4 \
    --optimizer-type  adamw \
    --loss-type  lsce \
    --lr-schedule cosine \
    --workers 16  \
