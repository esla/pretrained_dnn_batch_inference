#!/bin/bash

#sleep 2h;

python main.py \
	--learning_type='focal_loss'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/home/esla/datasets/training_for_grad_cam  	\
        --dataset_class_type='class_folders'	\
        --data_transform='data_transform4'	\
        --train_loss_idea='loss_idea2'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.01 	\
	--lr_scheduler='mtd4'     \
	--batch_size=128   \
	--net_type=vgg16	\
	--dropout=0	\
	--dataset=ba4	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
