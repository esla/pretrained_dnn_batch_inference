#!/bin/bash
#export netType='wide-resnet'

python main.py \
        --learning_type='multi_class'
	--datasets_class_folders_root_dir=/home/esla/research/datasets/dogs_vs_cats/dogs_vs_cats_dataset_class_folders	\
        --dataset_class_type='class folders'	\
	--lr=0.05	\
	--net_type=resnet18	\
	--dropout=0	\
	--dataset=isic2019	\
	--resume_from_model=''	\
	--inference_model=''	\
	--inference_dataset_dir=''	\
