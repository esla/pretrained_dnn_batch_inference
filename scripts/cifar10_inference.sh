#!/bin/bash

python main.py \
        --learning_type=multi_class	\
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
	--inference_model='/home/esla/research/isic-competition/my-workstation/image-classification-pytorch/checkpoint/may_16_18th/16-May-2020-10_59_17.714575-resnet18-cifar10-32/resnet18-epoch-222.t7'	  \
	--inference_dataset_dir='/home/esla/research/datasets/cifar10-pngs-in-folders/cifar10/train'	\
