#!/bin/bash

python main.py \
	--learning_type='focal_loss_target'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/home/esla/research/datasets/siim/exp_datasets/1024_dataset 	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform5'	\
    	--train_loss_idea='loss_idea1'	\
    	--temp_scale_idea='temp_scale_default'	\
	--lr=0.05	\
        --is_custom_lr=False	\
	--alpha=None	\
	--lr_scheduler='mtd5'     \
	--batch_size=32   \
	--net_type=resnet18	\
	--dropout=0	\
	--dataset=siim	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
