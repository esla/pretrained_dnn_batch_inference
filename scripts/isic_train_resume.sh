#!/bin/bash
#export netType='wide-resnet'
export netType='lenet'
export depth=28
export width=10
export dataset='isic2019'

python main.py \
    --learning_type='multi_class'	\
    --lr 0.001 \
    --net_type ${netType} \
    --depth ${depth} \
    --widen_factor ${width} \
    --dropout 0 \
    --dataset ${dataset} \
    --resume
