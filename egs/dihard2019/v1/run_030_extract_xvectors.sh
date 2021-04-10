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
use_gpu=false
xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

xvector_dir=exp/xvectors_diar/$nnet_name

if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in voxcelebcat 
    do
    	hyp_utils/xvectors/extract_xvectors_slidwin.sh --cmd "$xvec_cmd --mem 12G" --nj 300 ${xvec_args} \
	    --win-length 1.5 --win-shift 5 --snip-edges true --use-bin-vad true \
	    --feat-opts "--feat-frame-length 25 --feat-frame-shift 10" \
    	    $nnet data/${name}_combined \
    	    $xvector_dir/${name}_combined
    done
    exit
fi


if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in dihard2019_dev dihard2019_eval
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 10 ? $num_spk:10))
	hyp_utils/xvectors/extract_xvectors_slidwin.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --win-length 1.5 --win-shift 0.25 --write-timestamps true \
	    --feat-opts "--feat-frame-length 25 --feat-frame-shift 10" \
	    $nnet data/$name \
	    $xvector_dir/$name
    done
fi

exit
