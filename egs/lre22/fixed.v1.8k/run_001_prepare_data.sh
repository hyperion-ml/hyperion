#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh
stage=1

. parse_options.sh || exit 1;
. datapath.sh


if [ $stage -le 1 ];then
  # Prepares voxlingua 107 for training
  hyp_utils/conda_env.sh \
    local/prepare_voxlingua107.py \
    --corpus-dir $voxlingua_root \
    --output-dir data/voxlingua107 \
    --remove-langs en-en es-es ar-ar pt-pt \
    --map-langs-to-lre-codes \
    --target-fs 8000
  
fi

if [ $stage -le 2 ];then
  # Prepare LRE17 Training data
  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_train_root \
    --output-dir data/lre17_train \
    --subset train \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_train_root \
    --output-dir data/lre17_dev_cts \
    --subset dev \
    --source mls14 \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_train_root \
    --output-dir data/lre17_dev_afv \
    --subset dev \
    --source vast \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_eval_root \
    --output-dir data/lre17_eval_cts \
    --subset eval \
    --source mls14 \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_eval_root \
    --output-dir data/lre17_eval_afv \
    --subset eval \
    --source vast \
    --target-fs 8000

fi

if [ $stage -le 3 ];then
  hyp_utils/conda_env.sh \
    local/prepare_lre22_dev.py \
    --corpus-dir $lre22_dev_root \
    --output-dir data/lre22_dev \
    --target-fs 8000

fi

if [ $stage -le 4 ];then
  hyp_utils/conda_env.sh \
    local/prepare_lre22_eval.py \
    --corpus-dir $lre22_eval_root \
    --output-dir data/lre22_eval \
    --target-fs 8000

fi

if [ $stage -le 5 ];then
  local/download_lre22_scorer.sh
  local/download_focal.sh
fi
