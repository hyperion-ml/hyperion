#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 


if [ $stage -le 1 ];then
  # Prepare the dev portion of the VoxCeleb1 dev dataset.
  local/make_voxceleb1cat.pl $voxceleb1_root 16 data

  # Prepare the dev portion of the VoxCeleb2 dataset.
  local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
  local/make_voxceleb2cat.pl $voxceleb2_root test 16 data/voxceleb2cat_test

  utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train data/voxceleb2cat_test
  utils/fix_data_dir.sh data/voxcelebcat
fi

if [ $stage -le 2 ]; then
  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $mx6_root 16 data
  grep -v "trim 0.000 =0.000" data/mx6_mic/wav.scp > data/mx6_mic/wav.scp.tmp
  mv data/mx6_mic/wav.scp.tmp data/mx6_mic/wav.scp
  utils/fix_data_dir.sh data/mx6_mic
fi

if [ $stage -le 3 ];then
  # Prepare SITW dev to train x-vector
  local/make_sitw_train.sh $sitw_root dev 16 data/sitw_train_dev
  local/make_sitw_train.sh $sitw_root eval 16 data/sitw_train_eval
  utils/combine_data.sh data/sitw_train data/sitw_train_dev data/sitw_train_eval
fi

if [ $stage -le 4 ];then
  # Prepare voices
  local/make_voices19_challenge_dev.sh $voices_root ./data
  local/make_voices19_challenge_eval.sh $voices_root ./data
fi
