#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

feats_diar=`pwd -P`/exp/feats_diar
storage_name=sre19-v1-av-feats-diar-$(date +'%m_%d_%H_%M')
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;


if [ $stage -le 1 ];then

    for name in voxceleb sitw_dev_test sitw_eval_test \
	sre18_dev_test_vast sre18_eval_test_vast \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_test_core janus_eval_test_core 
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 40 ? $num_spk:40))
    	steps_kaldi_diar/prepare_feats.sh --nj $nj --cmd "$train_cmd" \
	    --storage_name $storage_name \
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
    for name in voxceleb sitw_dev_test sitw_eval_test \
        sre18_dev_test_vast sre18_eval_test_vast \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_test_core janus_eval_test_core 
    do
	num_spk=$(wc -l data_diar/${name}_cmn/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 40 ? $num_spk:40))
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	steps_kaldi_diar/vad_to_segments.sh --nj $nj --cmd "$train_cmd" \
	    data_diar/${name}_cmn data_diar/${name}_cmn_segmented
    done


fi

