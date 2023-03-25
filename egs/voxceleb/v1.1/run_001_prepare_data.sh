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
  local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
fi

if [ $stage -le 2 ];then
  # prepare voxceleb1 for test
  # This script is for the old version of the dataset
  # local/make_voxceleb1_oeh.pl $voxceleb1_root data
  # Use this for the newer version of voxceleb1:
  local/make_voxceleb1_v2_oeh.pl $voxceleb1_root data
fi

if [ $stage -le 3 ] && [ "$do_voxsrc22" == "true" ];then
  local/prepare_voxsrc22_dev.py \
    --vox1-corpus-dir $voxceleb1_root \
    --voxsrc22-corpus-dir $voxsrc22_root \
    --output-dir data/voxsrc22_dev
fi

if [ $stage -le 4 ] && [ "$do_voxsrc22" == "true" ];then
  local/prepare_voxsrc22_test.py \
    --corpus-dir $voxsrc22_root \
    --output-dir data/voxsrc22_test
fi

if [ $stage -le 5 ] && [ "$do_qmf" == "true" ];then
  # # split vox2 into 2 parts, for cohort and qmf training
  # utils/copy_data_dir.sh data/voxceleb2cat_train data/voxceleb2cat_train_odd
  # utils/copy_data_dir.sh data/voxceleb2cat_train data/voxceleb2cat_train_even
  # awk 'int(substr($2,3)) % 2 == 1' data/voxceleb2cat_train/utt2spk > data/voxceleb2cat_train_odd/utt2spk
  # utils/fix_data_dir.sh data/voxceleb2cat_train_odd
  # awk 'int(substr($2,3)) % 2 == 0' data/voxceleb2cat_train/utt2spk > data/voxceleb2cat_train_even/utt2spk
  # utils/fix_data_dir.sh data/voxceleb2cat_train_even
  # # we keep 3 utts per speaker
  # utils/subset_data_dir.sh --per-spk data/voxceleb2cat_train_odd 3 data/voxceleb2cat_train_subset_cohort
  # utils/subset_data_dir.sh --per-spk data/voxceleb2cat_train_even 3 data/voxceleb2cat_train_subset_qmf
  local/make_vox2_trials.py --data-dir data/voxceleb2cat_train
fi
