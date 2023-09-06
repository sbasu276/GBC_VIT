#!/bin/bash

python train_vit.py \
                --train_list="data/cls_split/train.txt" \
                --val_list="data/cls_split/val.txt" \
                --save_name="$1" \
                --lr=0.0001 \
                --optim="adam" \
                --batch_size=16 \
                --epochs=20 \
                --save_dir="$2" \
                --arch="$1"


