#!/bin/bash
#export netType='wide-resnet'
export netType='resnet18'
export depth=28
export width=10
export dataset='isic2019'

python main.py \
	--learning_type='multi_class'	\
	--input_image_size=32   \
	--datasets_class_folders_root_dir=/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10  	\
    	--dataset_class_type='class folders'	\
    	--data_transform='data_transform1'	\
	--batch_size=128    \
	--lr=0.1	\
	--lr_scheduler=mtd3     \
	--batch_size=128    \
	--net_type=resnet50	\
	--dropout=0	\
	--dataset=cifar10	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
