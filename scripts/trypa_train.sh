#!/bin/bash
#export netType='wide-resnet'

python main.py \
        --learning_type='multi_class'	\
        --input_image_size=224   \
	    --datasets_class_folders_root_dir=/home/esla/datasets/trypanosomes/experiment3	\
        --dataset_class_type='class folders'	\
	    --lr=0.00005	\
        --net_type=resnet18	\
        --dropout=0	\
        --dataset=ba4	\
        --resume_from_model=''	\
        --inference_model=''	\
        --inference_dataset_dir=''	\
