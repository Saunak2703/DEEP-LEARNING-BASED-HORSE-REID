#!/bin/bash
# ============================================================
# train_horse.sh — Horse Re-ID Training Script
# REID3 Dataset: 8 train horses, 5 test horses, 13 total
# ============================================================
# HOW TO RUN:
#   bash train_horse.sh
#
# DATASET STRUCTURE EXPECTED:
#   --root points to the folder CONTAINING the "horse" subfolder
#   So if your data is at:
#     C:\Users\LENOVO\OneDrive\Desktop\REID3\dataset\train\HORSE1\
#   Then --root should be:
#     C:\Users\LENOVO\OneDrive\Desktop\REID3\dataset
#   And the code expects:
#     {root}/horse/train/HORSE1/
#     {root}/horse/train/HORSE4/
#     ...
#     {root}/horse/test/HORSE0/
#     ...
#
# NOTE: Before running, create a symlink or copy your data so it matches:
#   REID3/dataset/horse/train/  (copy from REID3/dataset/train/)
#   REID3/dataset/horse/test/   (copy from REID3/dataset/test/)
# ============================================================

python main.py \
    -s horse \
    -t horse \
    --root "C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset" \
    --arch resnet50_fc512 \
    --max-epoch 60 \
    --train-batch-size 16 \
    --test-batch-size 64 \
    --optim adam \
    --lr 0.0003 \
    --stepsize 20 40 \
    --gamma 0.1 \
    --margin 0.3 \
    --num-instances 4 \
    --train-sampler RandomIdentitySampler \
    --label-smooth \
    --color-jitter \
    --random-erase \
    --height 256 \
    --width 128 \
    --eval-freq 10 \
    --start-eval 0 \
    --save-dir "C:/Users/LENOVO/OneDrive/Desktop/REID3/results/horse-resnet50-gpu" \
    --gpu-devices 0 \
    -j 0
