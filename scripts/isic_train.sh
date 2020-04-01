#!/bin/bash
#export netType='wide-resnet'
export netType='resnet18'
export depth=28
export width=10
export dataset='isic2019'

python main.py \
	--datasets_class_folders_root_dir=/home/esla/datasets/my-experiments/isic2019_nv_vs_rest	\
        --dataset_class_type='class folders'	\
	--lr=0.05	\
	--net_type=resnet50	\
	--dropout=0	\
	--dataset=isic2019	\
	--resume_from_model=/home/esla/research/isic-competition/my-workstation/image-classification-pytorch/checkpoint/isic2019/resnet-50-epoch-17.t7	\
	--inference_model=''	\
	--inference_dataset_dir=''	\

