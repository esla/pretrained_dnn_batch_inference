#!/bin/bash

python main.py \
        --validate_train_dataset	\
        --inference_only	\
        --inference_filename='testing_inference'	\
	--input_image_size=32   \
	--datasets_class_folders_root_dir=/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10  	\
    	--dataset_class_type='class folders'	\
    	--data_transform='data_transform4'	\
	--batch_size=32    \
	--dropout=0	\
	--dataset=cifar10	\
	--resume_from_model=''  \
	--inference_model='/media/esla/Samsung_T5/experiments_from_server/may_16_18th_used_train_dataset_for_validation/18-May-2020-02_45_54.846919-vgg11-cifar10-32/vgg11-epoch-61.t7'	  \
	--inference_dataset_dir='/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10/train'	\
