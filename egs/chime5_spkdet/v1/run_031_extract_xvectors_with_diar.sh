#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
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
    xvec_args="--chunk-length $xvec_chunk_length"
fi

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ]; then
    # Extracts x-vectors for evaluation
    for name in chime5_spkdet_test
    do
	name_out=${name}_${diar_name}
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors_from_wav_with_diar.sh \
	    --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name $diar_dir/$name/rttm \
	    $xvector_dir/$name_out
    done
fi
