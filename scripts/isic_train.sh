#!/bin/bash

#sleep 2h;

python main.py \
	--learning_type='focal_loss'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/home/esla/datasets/isic-dataset/isic-2019-7-classes/dataset  	\
        --dataset_class_type='class_folders'	\
        --data_transform='data_transform4'	\
        --train_loss_idea='loss_idea2'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.8 	\
	--lr_scheduler='mtd2'     \
	--batch_size=128   \
	--net_type=resnet18	\
	--dropout=0	\
	--dataset=isic2019	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
