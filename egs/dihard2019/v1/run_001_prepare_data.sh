#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
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

    # Prepare the VoxCeleb1 dataset for training.
    local/make_voxceleb1cat.pl $voxceleb1_root 16 data

    # Prepare the VoxCeleb2 dataset for training.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
    local/make_voxceleb2cat.pl $voxceleb2_root test 16 data/voxceleb2cat_test
    utils/combine_data.sh data/voxceleb2cat data/voxceleb2cat_train data/voxceleb2cat_test 
fi

if [ $stage -le 2 ];then
    # prepare Dihard2019
    local/make_dihard_2019_dev.sh $dihard2019_dev data/dihard2019_dev
    local/make_dihard_2019_dev.sh $dihard2019_eval data/dihard2019_eval
fi
