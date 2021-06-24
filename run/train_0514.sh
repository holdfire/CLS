#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8
MOBILE_MEMORY=20210514

OUTPUT=/home/projects/face_liveness/FAS-Transformer/ckpt/checkpoint_${MOBILE_MEMORY}/
mkdir -p ${OUTPUT}

LOG=/home/projects/face_liveness/FAS-Transformer/logs/logs_${MOBILE_MEMORY}
mkdir -p ${LOG}

python3 -u ./train.py  \
    --train-list  /home/projects/list/train_list/train_all_20210423_bbox.txt \
    --cfg home/projects/face_liveness/FAS-Transformer/models/configs/vit_custom_1.yaml \
    --arch deit \
    --input-size  224 \
    --crop-scale 2.5 \
    --resume /home/projects/face_liveness/FAS-Transformer/ckpt/checkpoint_20210514/epoch_1.pth.tar \
    --pth-save-dir  ${OUTPUT} \
    --log-save-dir ${LOG} \
    --pth-save-iter  10000 \
    --pretrained \
    --epochs  10 \
    --batch-size 256 \
    --learning-rate  5e-4 \
    --weight-decay 1e-4 \
    --optimizer-type  adamw \
    --loss-type  bce \
    --lr-schedule cosine \
    --workers 16  \
    --dist-url  tcp://localhost:10001  \
    --multiprocessing-distributed  \
    --world-size 1  \
    --rank 0  \
