#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the quiz dataset
# 2. Trains a inceptionv4 model on the quiz training set.
# 3. Evaluates the model on the quiz testing set.
#
# Usage:
# cd slim
# ./scripts/train_inceptionv4_on_quiz.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=~/temp/inception_v4-model

# Where the dataset is saved to.
DATASET_DIR=~/temp/quiz/

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=quiz \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=quiz \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
#  --preprocessing_name=cifarnet \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate=0.01 \
#  --learning_rate_decay_factor=0.1 \
#  --num_epochs_per_decay=200 \
#  --weight_decay=0.004 \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=quiz \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --batch_size=32 \
  --max_num_batches=128 \
  --eval_dir=~/temp/validation_eval_inception_v4