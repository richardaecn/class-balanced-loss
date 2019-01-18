#!/bin/sh

alias python='CUDA_VISIBLE_DEVICES=0 python3'

CIFAR_VERSION=10

LOSS_TYPE=softmax
python ./src/cifar_main.py --data-dir=./data/cifar-${CIFAR_VERSION}-data --job-dir=./results/cifar-${CIFAR_VERSION}/${LOSS_TYPE} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} || continue

LOSS_TYPE=sigmoid
python ./src/cifar_main.py --data-dir=./data/cifar-${CIFAR_VERSION}-data --job-dir=./results/cifar-${CIFAR_VERSION}/${LOSS_TYPE} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} || continue

LOSS_TYPE=focal
GAMMA=1.0
python ./src/cifar_main.py --data-dir=./data/cifar-${CIFAR_VERSION}-data --job-dir=./results/cifar-${CIFAR_VERSION}/${LOSS_TYPE}_${GAMMA} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} --gamma=${GAMMA}
