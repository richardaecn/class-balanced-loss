#!/bin/sh
# 1. update image mean and std in resnet_main.py.
# 2. update input image resolution resnet_preprocessing.py.

export PYTHONPATH="$PYTHONPATH:/home/yincui/tpu/models"

STORAGE_BUCKET=gs://tpu_training
DATASET=inat2017
DATA_DIR=${STORAGE_BUCKET}/data/${DATASET}

RESNET_DEPTH=50

NUM_TRAIN_IMAGES=579184
NUM_EVAL_IMAGES=95986
NUM_CLASSES=5089

TRAIN_STEPS=50905
TRAIN_BATCH_SIZE=1024
EVAL_BATCH_SIZE=1000
STEPS_PER_EVAL=2262
ITER_PER_LOOP=1131

# # 512 batch
# TRAIN_STEPS=101810
# TRAIN_BATCH_SIZE=512
# EVAL_BATCH_SIZE=500
# STEPS_PER_EVAL=4524
# ITER_PER_LOOP=1131

LR=0.4
LOG_STEPS=100
BETA=0.999
GAMMA=0.5

MODEL_DIR=${STORAGE_BUCKET}/${DATASET}_resnet${RESNET_DEPTH}_${BETA}_${GAMMA}
IMG_NUM_PER_CLASS_FILE=data/${DATASET}/img_num_per_cls.txt

python models/official/resnet/resnet_main.py \
  --tpu=${TPU_NAME} \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --resnet_depth=${RESNET_DEPTH} \
  --train_steps=${TRAIN_STEPS} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --eval_batch_size=${EVAL_BATCH_SIZE} \
  --num_train_images=${NUM_TRAIN_IMAGES} \
  --num_eval_images=${NUM_EVAL_IMAGES} \
  --num_label_classes=${NUM_CLASSES} \
  --steps_per_eval=${STEPS_PER_EVAL} \
  --iterations_per_loop=${ITER_PER_LOOP} \
  --base_learning_rate=${LR} \
  --log_step_count_steps=${LOG_STEPS} \
  --beta=${BETA} \
  --gamma=${GAMMA} \
  --img_num_per_cls_file=${IMG_NUM_PER_CLASS_FILE}
