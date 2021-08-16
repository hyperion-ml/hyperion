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
    local/make_voxceleb1cat_v2.pl $voxceleb1_root 16 data

    # Prepare the VoxCeleb2 dataset.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
    local/make_voxceleb2cat.pl $voxceleb2_root test 16 data/voxceleb2cat_test

    utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train data/voxceleb2cat_test
    utils/fix_data_dir.sh data/voxcelebcat
fi

if [ $stage -le 2 ];then
  # Prepare SRE CTS superset
  hyp_utils/conda_env.sh \
    local/prepare_sre_cts_superset.py \
    --data-dir $sre_superset_root \
    --target-fs 16000 \
    --output-dir data/sre_cts_superset_16k
fi

exit
