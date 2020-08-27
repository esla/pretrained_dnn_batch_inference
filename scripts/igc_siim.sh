#!/bin/bash

#sleep 2h;
#--datasets_class_folders_root_dir=/media/esla/DStorage/DataSets/siim_resized/cropped_from_1024/dataset_seed1337  	\
python main.py \
	--learning_type='focal_loss_target'	    \
	--input_image_size=224   \
    	--datasets_class_folders_root_dir=/media/esla/DStorage/DataSets/siim_resized/512	\
    	--dataset_class_type='class_folders'	\
    	--data_transform='data_transform4'	\
    	--train_loss_idea='loss_idea2'	\
    	--temp_scale_idea='temp_scale_default'	\
	--lr=0.8 	\
	--lr_scheduler='mtd2'     \
	--batch_size=32   \
	--net_type=efficient-b3	\
	--dropout=0	\
	--dataset=siim	\
	--resume_from_model=''  \
	--inference_model=''	\
	--inference_dataset_dir=''	\
