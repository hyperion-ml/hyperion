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
    xvec_cmd="$cuda_eval_cmd --mem 4G"
else
    xvec_cmd="$train_cmd --mem 6G"
fi

plda_num_augs=$diar_plda_num_augs

xvector_dir=exp/xvectors_diar/$nnet_name

if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in voxcelebcat
    do
	if [ $plda_num_augs -eq 0 ]; then
    	    hyp_utils/xvectors/extract_xvectors_slidwin_from_wav.sh \
		--cmd "$xvec_cmd" --nj 100 ${xvec_args} \
		--win-length 1.5 --win-shift 5 --snip-edges true --use-bin-vad true \
		--feat-config $feat_config \
    		$nnet data/${name} \
    		$xvector_dir/${name}
	else
	    hyp_utils/xvectors/extract_xvectors_slidwin_from_wav.sh --cmd "$xvec_cmd" --nj 300 ${xvec_args} \
		--win-length 1.5 --win-shift 5 --snip-edges true --use-bin-vad true \
		--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    		$nnet data/${name} \
    		$xvector_dir/${name}_augx${plda_num_augs} \
		data/${name}_augx${plda_num_augs}
	fi
    done
fi


if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in sitw_dev_test sitw_eval_test \
	sre18_eval_test_vast sre18_dev_test_vast \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_test_core janus_eval_test_core
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	hyp_utils/xvectors/extract_xvectors_slidwin_from_wav.sh \
	    --win-length 1.5 --win-shift 0.25 --write-timestamps true \
	    --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/$name
    done
fi

exit
