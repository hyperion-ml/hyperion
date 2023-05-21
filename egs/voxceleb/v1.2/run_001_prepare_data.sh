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
. $config_file

if [ $stage -le 1 ];then
  # Prepare the VoxCeleb2 dataset for training.
  hyp_utils/conda_env.sh \
    prepare_data.py voxceleb2 --subset dev --corpus-dir $voxceleb2_root \
    --cat-videos --use-kaldi-ids \
    --output-dir data/voxceleb2cat_train
  #local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
fi
exit
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

# if [ $stage -le 4 ] && [ "$do_voxsrc22" == "true" ];then
#   local/prepare_voxsrc22_test.py \
#     --corpus-dir $voxsrc22_root \
#     --output-dir data/voxsrc22_test
# fi

if [ $stage -le 5 ] && [ "$do_qmf" == "true" ];then
  # # split vox2 into 2 parts, for cohort and qmf training
  local/make_vox2_trials.py --data-dir data/voxceleb2cat_train
fi
