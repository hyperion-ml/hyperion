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
ft=0
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
    xvec_args="--chunk-length $xvec_chunk_length"
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
    for name in voxcelebcat 
    do
	if [ $plda_num_augs -eq 0 ]; then
    	    steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd --mem 12G" --nj 100 ${xvec_args} \
		--random-utt-length true --min-utt-length 400 --max-utt-length 14000 \
		--feat-config $feat_config \
    		$nnet data/${name} \
    		$xvector_dir/${name}
	else
	    steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd --mem 12G" --nj 300 ${xvec_args} \
		--random-utt-length true --min-utt-length 400 --max-utt-length 14000 \
		--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    		$nnet data/${name} \
    		$xvector_dir/${name}_augx${plda_num_augs} \
		data/${name}_augx${plda_num_augs}
	fi
    done
fi

if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in chime5_spkdet_enroll chime5_spkdet_test chime5_spkdet_test_gtvad
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/$name
    done
fi

if [ $stage -le 3 ]; then
    mkdir -p $xvector_dir/chime5_spkdet_gtvad
    cat $xvector_dir/chime5_spkdet_{enroll,test_gtvad}/xvector.scp > $xvector_dir/chime5_spkdet_gtvad/xvector.scp

    mkdir -p $xvector_dir/chime5_spkdet
    cat $xvector_dir/chime5_spkdet_{enroll,test}/xvector.scp > $xvector_dir/chime5_spkdet/xvector.scp
fi


exit
