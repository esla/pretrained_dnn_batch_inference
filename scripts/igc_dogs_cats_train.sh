#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
	--learning_type='focal_loss_target'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/media/esla/DStorage/DataSets/dogs_vs_cats/dogs_vs_cats_dataset_class_folders  	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform5'	\
        --train_loss_idea='loss_idea1'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.1	\
        --lr_scheduling_mtd='not_custom'	\
	--alpha=None	\
	--lr_scheduler='mtd2'     \
	--batch_size=32    \
	--net_type=resnet18	\
	--dropout=0	\
	--dataset=dogs_cats	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir='/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10/train'	\
