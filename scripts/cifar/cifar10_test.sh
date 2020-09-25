#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
	--learning_type='multi_class'	\
	--input_image_size=32   \
	--datasets_class_folders_root_dir=/home/esla/Documents/research_related/dataset/cifar10-pngs-in-folders/cifar10  	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform1'	\
        --train_loss_idea='loss_idea1'	\
        --temp_scale_idea='temp_scale_default'	\
	--lr=0.4	\
        --lr_scheduling_mtd='non-custom'	\
	--alpha=None	\
	--lr_scheduler='mtd2'     \
	--batch_size=64    \
	--net_type=resnext29_2x64d	\
	--dropout=0	\
	--dataset=cifar10	\
	--inference_only  \
	--inference_model='/home/esla/Documents/research_related/useful_research_implementations_n_tools/my-git-repos/image-classification-pytorch/checkpoint/16-Sep-2020-11_48_26.528_resnext29_2x64d_cifar10_32/resnext29_2x64d-epoch-182.pth'	\
	--inference_dataset_dir='/home/esla/Documents/research_related/dataset/cifar10-pngs-in-folders/cifar10/val'	\
