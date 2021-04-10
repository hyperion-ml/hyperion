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

    # Prepare the VoxCeleb1 dataset for training.
    local/make_voxceleb1cat.pl $voxceleb1_root 16 data

    # Prepare the VoxCeleb2 dataset for training.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
    utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train
fi

if [ $stage -le 2 ];then
    # prepare Dihard2019
    local/make_dihard2019.sh $dihard2019_dev data/dihard2019_dev
    local/make_dihard2019.sh $dihard2019_eval data/dihard2019_eval
fi
