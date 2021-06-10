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
    local/make_voxceleb1_o.pl $voxceleb1_root data
    local/make_trials_subset.py --in-key-file data/voxceleb1_test/trials_o_clean --out-key-file data/voxceleb1_test/trials_o_clean_1000_1000 --ntar 1000 --nnon 1000
fi



