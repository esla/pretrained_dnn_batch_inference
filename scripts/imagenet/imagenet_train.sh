#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 
python main.py \
	--learning_type='focal_loss_target'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/home/esla/datasets/imagenet/ILSVRC2015/Data/CLS-LOC  	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform4'	\
        --train_loss_idea='loss_idea1'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.1	\
        --lr_scheduling_mtd='non-custom'	\
	--alpha=None	\
	--lr_scheduler='mtd2'     \
	--batch_size=512    \
	--net_type=resnet50	\
	--dropout=0	\
	--dataset=imagenet	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
