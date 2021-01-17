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

. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length 12800"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

xvector_dir=exp/xvectors/$ft_nnet_name

if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in $plda_data
    do
    	steps_xvec/extract_xvectors.sh --cmd "$xvec_cmd --mem 12G" --nj 300 ${xvec_args} \
	    --random-utt-length true --min-utt-length 400 --max-utt-length 14000 \
    	    $ft_nnet data/${name}_combined \
    	    $xvector_dir/${name}_combined
    done
fi

if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in voxceleb1_test 
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
					      $ft_nnet data/$name \
					      $xvector_dir/$name
    done
fi

exit
