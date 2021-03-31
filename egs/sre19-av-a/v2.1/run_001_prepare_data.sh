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
    # Make SITW dev and eval sets
    local/make_sitw.sh $sitw_root 16 data/sitw
fi

if [ $stage -le 3 ];then
    # Prepare sre18
    local/make_sre18_dev.sh $sre18_dev_root 16 data
    local/make_sre18_eval.sh $sre18_eval_root 16 data
fi

if [ $stage -le 4 ];then
    # Prepare sre19
    local/make_sre19av_a_dev.sh $sre19_dev_root 16 data
    local/make_sre19av_a_eval.sh $sre19_eval_root 16 data
fi

if [ $stage -le 5 ];then
    local/make_janus_core.sh $janus_root 16 data
fi

if [ $stage -le 6 ];then
    local/make_dihard_train.sh $dihard2_dev dev data
    local/make_dihard_train.sh $dihard2_eval eval data
fi

exit
