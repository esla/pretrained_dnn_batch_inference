#!/bin/bash
#export netType='wide-resnet'

python main.py \
	--learning_type='multi_class'	\
	--input_image_size=32   \
	--datasets_class_folders_root_dir=/media/esla/DStorage/DataSets/cifar100png  	\
	--data_transform=data_transform1	\
    	--dataset_class_type='class folders'	\
	--lr=0.1	\
	--lr_scheduler=mtd3	\
	--batch_size=128    \
	--net_type=efficientnetB0	\
	--dropout=0	\
	--dataset=cifar100	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
