#!/bin/bash

#sleep 2h;

python main.py \
	--learning_type='focal_loss_target'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/home/esla/research/datasets/siim/exp_datasets/cropped_from_1024/dataset_seed1337  	\
        --dataset_class_type='class_folders'	\
        --data_transform='data_transform5'	\
        --train_loss_idea='loss_idea2'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.8	\
	--alpha=0.17	\
	--lr_scheduler='mtd2'     \
	--batch_size=128   \
	--net_type=resnet18	\
	--dropout=0	\
	--dataset=siim	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
