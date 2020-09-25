#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main-ts.py \
	--learning_type='multi_class'	\
	--input_image_size=32   \
	--datasets_class_folders_root_dir=/home/esla/Documents/research_related/dataset/cifar10-pngs-in-folders/cifar10  	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform1'	\
        --train_loss_idea='loss_idea1'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.4	\
        --lr_scheduling_mtd='non-custom'	\
	--alpha=None	\
	--lr_scheduler='mtd2'     \
	--batch_size=64    \
	--net_type=resnext29_2x64d	\
	--dropout=0	\
	--dataset=cifar10	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir='/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10/train'	\
