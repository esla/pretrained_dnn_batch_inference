#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 
python main.py \
	--learning_type='multi_class'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/home/esla/datasets/imagenet/subset-10  	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform4'	\
        --train_loss_idea='loss_idea1'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.6	\
        --lr_scheduling_mtd='non_custom'	\
	--alpha=None	\
	--lr_scheduler='mtd2'     \
	--batch_size=32    \
	--net_type=resnext101_32x8d	\
	--dropout=0	\
	--dataset=imagenet	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
