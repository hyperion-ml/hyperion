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
  # Prepare the VoxCeleb2 dataset for training.
  local/make_voxceleb2.pl $voxceleb2_root dev 16 data/voxceleb2_train
fi

if [ $stage -le 2 ];then
  # prepare voxceleb1 for test
  # This script is for the old version of the dataset and NOT processing LANG,
  # GENDER, NAT
  local/make_voxceleb1_oldversion_oeh.pl $voxceleb1_root data
  # This script is for the old version of the dataset:
  # local/make_voxceleb1_oeh.pl $voxceleb1_root data
  # Use this for the newer version of voxceleb1:
  # local/make_voxceleb1_v2_oeh.pl $voxceleb1_root data
fi
