#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

feats_diar=`pwd -P`/exp/feats_diar
storage_name=sre18-v1.8k-diar-$(date +'%m_%d_%H_%M')
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;


if [ $stage -le 1 ];then

    for name in voxceleb sitw_dev_test sitw_eval_test sre18_eval_test_vast; do
    	steps_kaldi_diar/prepare_feats.sh --nj 40 --cmd "$train_cmd" --storage_name $storage_name \
    					  data/$name data_diar/${name}_cmn $feats_diar/${name}_cmn
    	cp data/$name/vad.scp data_diar/${name}_cmn/
    	if [ -f data/$name/segments ]; then
    	    cp data/$name/segments data_diar/${name}_cmn/
    	fi
    	utils/fix_data_dir.sh data_diar/${name}_cmn
    done

    for name in sre18_dev_test_vast; do
    	steps_kaldi_diar/prepare_feats.sh --nj 1 --cmd "$train_cmd" --storage_name $storage_name \
    					  data/$name data_diar/${name}_cmn $feats_diar/${name}_cmn
    	cp data/$name/vad.scp data_diar/${name}_cmn/
    	if [ -f data/$name/segments ]; then
    	    cp data/$name/segments data_diar/${name}_cmn/
    	fi
    	utils/fix_data_dir.sh data_diar/${name}_cmn
    done

fi


if [ $stage -le 2 ];then
    # Create segments to extract x-vectors
    for name in voxceleb sitw_dev_test sitw_eval_test sre18_eval_test_vast
    do
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	steps_kaldi_diar/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
					    data_diar/${name}_cmn data_diar/${name}_cmn_segmented
    done

    for name in sre18_dev_test_vast
    do
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	steps_kaldi_diar/vad_to_segments.sh --nj 1 --cmd "$train_cmd" \
					    data_diar/${name}_cmn data_diar/${name}_cmn_segmented
    done

fi

