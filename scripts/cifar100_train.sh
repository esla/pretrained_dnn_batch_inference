#!/bin/bash
export netType='wide-resnet'
export depth=28
export width=10
export dataset='cifar100'

python main.py \
    --lr 0.5 \
    --net_type ${netType} \
    --depth ${depth} \
    --widen_factor ${width} \
    --dropout 0.3 \
    --dataset ${dataset}
