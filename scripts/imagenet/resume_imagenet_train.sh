#!/bin/bash

python main.py \
	--learning_type='multi_class'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=/home/esla/datasets/imagenet/ILSVRC2015/Data/CLS-LOC  	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform2'	\
        --train_loss_idea='loss_idea4'	\
        --temp_scale_idea='temp_scale_idea3'	\
	--lr=0.8	\
	--lr_scheduler=mtd2     \
	--batch_size=256    \
	--net_type=resnet50	\
	--dropout=0	\
	--dataset=imagenet	\
        --resume_training	\
	--resume_from_model='/home/esla/research/isic-competition/my-workstation/image-classification-pytorch/checkpoint/22-May-2020-13_14_16.768517-resnet50-imagenet-224/resnet50-epoch-1.t7'  \
