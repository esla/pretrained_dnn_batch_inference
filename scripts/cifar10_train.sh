#!/bin/bash

python main.py \
	--learning_type='focal_loss'	\
	--input_image_size=32   \
	--datasets_class_folders_root_dir=/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10  	\
    	--dataset_class_type='class folders'	\
    	--data_transform='data_transform4'	\
        --train_loss_idea='loss_idea4'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.8	\
	--lr_scheduler=mtd2     \
	--batch_size=64    \
	--net_type=resnext29_2x64d	\
	--dropout=0	\
	--dataset=cifar10	\
	--resume_from_model=''  \
	--inference_model='/home/esla/research/isic-competition/my-workstation/image-classification-pytorch/checkpoint/15-May-2020-07_42_05.217089-resnet18-cifar10-32/resnet18-epoch-130.t7'	\
	--inference_dataset_dir='/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10/train'	\
