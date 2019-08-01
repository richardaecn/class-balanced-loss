#!/bin/sh

alias python='CUDA_VISIBLE_DEVICES=1 python3'

CIFAR_VERSION=10
IM_FACTOR=0.02
BETA=0.9999

LOSS_TYPE=softmax
python ./src/cifar_main.py --data-dir=./data/cifar-${CIFAR_VERSION}-data-im-${IM_FACTOR} --job-dir=./results/cifar-${CIFAR_VERSION}-im-${IM_FACTOR}/${LOSS_TYPE}_${BETA} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} --imb-factor=${IM_FACTOR} --learning-rate-multiplier 1 0.01 0.0001 --beta=${BETA}|| continue

LOSS_TYPE=sigmoid
python ./src/cifar_main.py --data-dir=./data/cifar-${CIFAR_VERSION}-data-im-${IM_FACTOR} --job-dir=./results/cifar-${CIFAR_VERSION}-im-${IM_FACTOR}/${LOSS_TYPE}_${BETA} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} --imb-factor=${IM_FACTOR} --learning-rate-multiplier 1 0.01 0.0001 --beta=${BETA}|| continue

LOSS_TYPE=focal
GAMMA=1.0
python ./src/cifar_main.py --data-dir=./data/cifar-${CIFAR_VERSION}-data-im-${IM_FACTOR} --job-dir=./results/cifar-${CIFAR_VERSION}-im-${IM_FACTOR}/${LOSS_TYPE}_${GAMMA}_${BETA} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} --imb-factor=${IM_FACTOR} --learning-rate-multiplier 1 0.01 0.0001 --beta=${BETA} --gamma=${GAMMA}
