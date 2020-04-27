#!/bin/bash
#export netType='wide-resnet'
export netType='resnet18'
export depth=28
export width=10
export dataset='isic2019'

python main.py \
	--learning_type='multi_class'	\
	--datasets_class_folders_root_dir=/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10  	\
        --dataset_class_type='class folders'	\
	--lr=0.1	\
	--net_type=lenet	\
	--dropout=0	\
	--dataset=cifar10	\
	--resume_from_model=/home/esla/research/isic-competition/my-workstation/image-classification-pytorch/checkpoint/cifar10_ce_bs-4096_lr-0.01_momen-0.9_wd-1e-2/lenet-epoch-252.t7	\
	--inference_model=''	\
	--inference_dataset_dir=''	\
