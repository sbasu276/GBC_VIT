#!/bin/bash

python test.py \
	--img_dir="../XGC-data/xgc" \
	--val_list="../XGC-data/xgc-cls.txt" \
	--model_file="model_weights/radformer/radformer.pkl" \
	--pred_name="radformer.csv"
