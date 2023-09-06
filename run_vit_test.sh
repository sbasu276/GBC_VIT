#!/bin/bash

python test_vit.py \
	--img_dir="../XGC-data/xgc" \
	--val_list="../XGC-data/xgc-cls.txt" \
	--model_file="$2" \
	--arch="$1" \
	--pred_name="$1.csv"
