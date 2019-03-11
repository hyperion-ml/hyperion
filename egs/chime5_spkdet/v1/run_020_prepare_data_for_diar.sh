#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

feats_diar=`pwd -P`/exp/feats_diar
storage_name=chime5-spkdet-v1-diar-$(date +'%m_%d_%H_%M')
stage=1

. parse_options.sh || exit 1;


if [ $stage -le 1 ];then

#    for name in voxceleb chime5_spkdet_test; do
    for name in chime5_spkdet_test; do
    	steps_kaldi_diar/prepare_feats.sh --nj 39 --cmd "$train_cmd" --storage_name $storage_name \
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
    for name in voxceleb chime5_spkdet_test
    do
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	steps_kaldi_diar/vad_to_segments.sh --nj 39 --cmd "$train_cmd" \
					    data_diar/${name}_cmn data_diar/${name}_cmn_segmented
    done
    exit
fi

