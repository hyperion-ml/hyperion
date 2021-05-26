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
    # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
    # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
    # contains the speakers that overlap between VoxCeleb1 and our evaluation
    # set SITW.  The script removes these overlapping speakers from VoxCeleb1.
    local/make_voxceleb1cat.pl $voxceleb1_root 16 data

    # Prepare the dev portion of the VoxCeleb2 dataset.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train
    local/make_voxceleb2cat.pl $voxceleb2_root test 16 data/voxceleb2cat_test

    utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train data/voxceleb2cat_test
    utils/fix_data_dir.sh data/voxcelebcat
fi

if [ $stage -le 2 ];then
    # Prepare chime5
    local/make_chime5_spkdet.sh $chime5_root ./data
fi

#some additional training sets
# if [ $stage -le 3 ];then
#   # Prepare SITW dev to train x-vector
#     local/make_sitw_train.sh $sitw_root dev 16 data/sitw_train_dev
#     local/make_sitw_train.sh $sitw_root eval 16 data/sitw_train_eval
#     utils/combine_data.sh data/sitw_train data/sitw_train_dev data/sitw_train_eval
# fi

# if [ $stage -le 4 ]; then
#     # Prepare telephone and microphone speech from Mixer6.
#     local/make_mx6.sh $mx6_root 16 data
#     grep -v "trim 0.000 =0.000" data/mx6_mic/wav.scp > data/mx6_mic/wav.scp.tmp
#     mv data/mx6_mic/wav.scp.tmp data/mx6_mic/wav.scp
#     fix_data_dir.sh data/mx6_mic
#     exit
# fi

exit
