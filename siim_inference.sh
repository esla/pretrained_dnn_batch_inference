#!/bin/bash

#sleep 2h;

python main.py \
	--learning_type='focal_loss_target'	\
	--input_image_size=224   \
	--datasets_class_folders_root_dir=''  	\
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
	--inference_model=/media/esla/DataDrive/ghent_phd_research/git-repos-igc/image-classification-pytorch/checkpoint/07-Aug-2020-01_31_18.473009-resnext50_32x4d-siim-224/resnext50_32x4d-epoch-99.pth	\
	--inference_dataset_dir=/media/esla/DStorage/DataSets/siim_resized/1024/test	\
