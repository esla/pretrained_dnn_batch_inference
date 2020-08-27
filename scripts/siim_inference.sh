#!/bin/bash

#sleep 2h;

python inference.py \
	--input_image_size=224   \
	--datasets_class_folders_root_dir=''  	\
    	--dataset_class_type='class_folders'	\
	--dataset=siim	\
	--inference_model=$1	\
	--inference_dataset_dir=/home/esla/research/datasets/siim/jpeg/test_resized/test_orig_1024	\
