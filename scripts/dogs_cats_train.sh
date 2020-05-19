#!/bin/bash

python main.py \
	--learning_type='multi_class'	\
	--input_image_size=32   \
	--datasets_class_folders_root_dir=/home/esla/research/datasets/dogs_vs_cats/dogs_vs_cats_dataset_class_folders  	\
    	--dataset_class_type='class folders'	\
    	--data_transform='data_transform3'	\
	--lr=0.8	\
	--lr_scheduler=mtd2     \
	--batch_size=32    \
	--net_type=lenet	\
	--dropout=0	\
	--dataset=cifar10	\
	--resume_from_model=''  \
	--inference_model='/home/esla/research/isic-competition/my-workstation/image-classification-pytorch/checkpoint/15-May-2020-07_42_05.217089-resnet18-cifar10-32/resnet18-epoch-130.t7'	\
	--inference_dataset_dir='/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10/train'	\
