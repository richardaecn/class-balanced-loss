#!/bin/sh
# chmod +x ./cifar_im_trainval_resample_v2.sh
# ./cifar_im_trainval_resample_v2.sh 0 10
# ./cifar_im_trainval_resample_v2.sh 1 100


GPU_NUM=$1
CIFAR_VERSION=$2

if [[ -z "${GPU_NUM}" ]]
then
    echo "GPU_NUM is empty, use default 0"
    GPU_NUM=0
fi

if [[ -z "${CIFAR_VERSION}" ]]
then
    echo "CIFAR_VERSION is empty, use 10 as default"
	CIFAR_VERSION=10
fi


IM_FACTOR=0.02
ROOT_DIR=/media/user/2tb/2018FocalLoss

for BETA in 0.9; do
	LOSS_TYPE=softmax
	CUDA_VISIBLE_DEVICES=${GPU_NUM} python3 ./src/cifar_main.py --data-dir=${ROOT_DIR}/data/cifar-${CIFAR_VERSION}-data-im-${IM_FACTOR} --job-dir=${ROOT_DIR}/results/cifar-${CIFAR_VERSION}-im-${IM_FACTOR}/${LOSS_TYPE}_${BETA} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} --imb-factor=${IM_FACTOR} --learning-rate-multiplier 1 0.01 0.0001 --beta=${BETA} --is-resample || continue

	LOSS_TYPE=sigmoid
	CUDA_VISIBLE_DEVICES=${GPU_NUM} python3 ./src/cifar_main.py --data-dir=${ROOT_DIR}/data/cifar-${CIFAR_VERSION}-data-im-${IM_FACTOR} --job-dir=${ROOT_DIR}/results/cifar-${CIFAR_VERSION}-im-${IM_FACTOR}/${LOSS_TYPE}_${BETA} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} --imb-factor=${IM_FACTOR} --learning-rate-multiplier 1 0.01 0.0001 --beta=${BETA} --is-resample || continue

	LOSS_TYPE=focal
	GAMMA=0.5
	CUDA_VISIBLE_DEVICES=${GPU_NUM} python3 ./src/cifar_main.py --data-dir=${ROOT_DIR}/data/cifar-${CIFAR_VERSION}-data-im-${IM_FACTOR} --job-dir=${ROOT_DIR}/results/cifar-${CIFAR_VERSION}-im-${IM_FACTOR}/${LOSS_TYPE}_${GAMMA}_${BETA} --data-version=${CIFAR_VERSION} --loss-type=${LOSS_TYPE} --imb-factor=${IM_FACTOR} --learning-rate-multiplier 1 0.01 0.0001 --beta=${BETA} --gamma=${GAMMA} --is-resample
done
