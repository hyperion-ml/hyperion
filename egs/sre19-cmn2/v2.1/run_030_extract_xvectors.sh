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
ft=0
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
    nnet=$ft_nnet
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
    nnet=$ft2_nnet
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
    nnet=$ft3_nnet
fi

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in sre_tel
    do
	if [ $plda_num_augs -eq 0 ];then
    	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj 100 ${xvec_args} \
		--random-utt-length true --min-utt-length 1000 --max-utt-length 6000 \
		--feat-config $feat_config \
    		$nnet data/${name} \
    		$xvector_dir/${name}
	else
	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj 100 ${xvec_args} \
		--random-utt-length true --min-utt-length 1000 --max-utt-length 6000 \
		--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    		$nnet data/${name} \
    		$xvector_dir/${name}_augx${plda_num_augs} \
		data/${name}_augx${plda_num_augs}
	fi
    done
fi

if [ $stage -le 2 ]; then
    # Extract xvectors for adapting LDA/PLDA
    for name in sre18_cmn2_adapt_lab
    do
	if [ $plda_num_augs -eq 0 ];then
    	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj 30 ${xvec_args} \
		--feat-config $feat_config \
    		$nnet data/${name} \
    		$xvector_dir/${name}
	else
	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj 100 ${xvec_args} \
		--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    		$nnet data/${name} \
    		$xvector_dir/${name}_augx${plda_num_augs} \
		data/${name}_augx${plda_num_augs}
	fi
    done
fi


if [ $stage -le 3 ]; then
    # Extracts x-vectors for evaluation
    for name in sre18_dev_unlabeled \
		    sre18_eval40_enroll_cmn2 sre18_eval40_test_cmn2 \
		    sre19_eval_enroll_cmn2 sre19_eval_test_cmn2
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/$name
    done
fi


if [ $stage -le 4 ]; then
    mkdir -p $xvector_dir/sre18_eval40_cmn2
    cat $xvector_dir/sre18_eval40_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_eval40_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre19_eval_cmn2
    cat $xvector_dir/sre19_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre19_eval_cmn2/xvector.scp
fi


exit
