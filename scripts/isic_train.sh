#!/bin/bash
#export netType='wide-resnet'
export netType='resnet18'
export depth=28
export width=10
export dataset='isic2019'

python main.py \
    --lr 0.01 \
    --net_type ${netType} \
    --depth ${depth} \
    --widen_factor ${width} \
    --dropout 0 \
    --dataset ${dataset} \
