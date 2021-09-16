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
    # Prepare the VoxCeleb1 dataset.  
    local/make_voxceleb1cat_v2.pl $voxceleb1_root 8 data

    # Prepare the VoxCeleb2 dataset.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 8 data/voxceleb2cat_train
    local/make_voxceleb2cat.pl $voxceleb2_root test 8 data/voxceleb2cat_test

    utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train data/voxceleb2cat_test
    utils/fix_data_dir.sh data/voxcelebcat

fi

if [ $stage -le 2 ];then
  # Prepare SRE CTS superset
  hyp_utils/conda_env.sh \
    local/prepare_sre_cts_superset.py \
    --corpus-dir $sre_superset_root \
    --target-fs 8000 \
    --output-dir data/sre_cts_superset_8k

  hyp_utils/conda_env.sh \
    local/trn_dev_split_sre_cts_superset.py \
    --input-dir data/sre_cts_superset_8k \
    --trn-dir data/sre_cts_superset_8k_trn \
    --dev-dir data/sre_cts_superset_8k_dev \
    --num-dev-spks-cmn 66 \
    --num-dev-spks-yue 34 

fi

if [ $stage -le 3 ];then
  # Prepare SRE16 dev for training
  local/make_sre16_train_dev.sh $sre16_dev_root 8 data
  # Prepare SRE16 Eval
  # 60% for training 40% for evaluation/calibration
  local/make_sre16_eval_tr60_ev40.sh $sre16_eval_root 8 data
fi

if [ $stage -le 4 ];then
  # Prepare SRE21 dev
  hyp_utils/conda_env.sh \
    local/prepare_sre21av_dev_audio.py \
    --corpus-dir $sre21_dev_root \
    --target-fs 8000 \
    --output-path data/sre21_audio_dev \
    --av-output-path data/sre21_audio-visual_dev
  
fi

if [ $stage -le 5 ];then
  # Prepare SRE21 eval
  hyp_utils/conda_env.sh \
    local/prepare_sre21av_eval_audio.py \
    --corpus-dir $sre21_eval_root \
    --target-fs 8000 \
    --output-path data/sre21_audio_eval \
    --av-output-path data/sre21_audio-visual_eval
  
fi

exit
