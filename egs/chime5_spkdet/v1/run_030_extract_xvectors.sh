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
    # Extracts x-vectors dihard cohort
    for name in dihard2_train
    do
	nj=200
	steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/$name
    done
fi

if [ $stage -le 3 ]; then
    # Extracts x-vectors for evaluation
    for name in sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test \
	sre18_eval_enroll_vast sre18_eval_test_vast sre18_dev_enroll_vast sre18_dev_test_vast \
	sre19_av_a_dev_enroll sre19_av_a_eval_enroll \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_enroll janus_dev_test_core janus_eval_enroll janus_eval_test_core 
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
    
    utils/combine_data.sh data/sitw_dev data/sitw_dev_enroll data/sitw_dev_test
    mkdir -p $xvector_dir/sitw_dev
    cat $xvector_dir/sitw_dev_{enroll,test}/xvector.scp > $xvector_dir/sitw_dev/xvector.scp

    mkdir -p $xvector_dir/sitw_eval
    cat $xvector_dir/sitw_eval_{enroll,test}/xvector.scp > $xvector_dir/sitw_eval/xvector.scp

    utils/combine_data.sh data/sre18_dev_vast data/sre18_dev_enroll_vast data/sre18_dev_test_vast 
    mkdir -p $xvector_dir/sre18_dev_vast
    cat $xvector_dir/sre18_dev_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_dev_vast/xvector.scp

    utils/combine_data.sh data/sre18_eval_vast data/sre18_eval_enroll_vast data/sre18_eval_test_vast 
    mkdir -p $xvector_dir/sre18_eval_vast
    cat $xvector_dir/sre18_eval_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_eval_vast/xvector.scp

    utils/combine_data.sh data/sre19_av_a_dev data/sre19_av_a_dev_enroll data/sre19_av_a_dev_test 
    mkdir -p $xvector_dir/sre19_av_a_dev
    cat $xvector_dir/sre19_av_a_dev_{enroll,test}/xvector.scp > $xvector_dir/sre19_av_a_dev/xvector.scp

    utils/combine_data.sh data/sre19_av_a_eval data/sre19_av_a_eval_enroll data/sre19_av_a_eval_test 
    mkdir -p $xvector_dir/sre19_av_a_eval
    cat $xvector_dir/sre19_av_a_eval_{enroll,test}/xvector.scp > $xvector_dir/sre19_av_a_eval/xvector.scp

    utils/combine_data.sh data/janus_dev_core data/janus_dev_enroll data/janus_dev_test_core
    mkdir -p $xvector_dir/janus_dev_core
    cat $xvector_dir/janus_dev_{enroll,test_core}/xvector.scp > $xvector_dir/janus_dev_core/xvector.scp

    utils/combine_data.sh data/janus_eval_core data/janus_eval_enroll data/janus_eval_test_core
    mkdir -p $xvector_dir/janus_eval_core
    cat $xvector_dir/janus_eval_{enroll,test_core}/xvector.scp > $xvector_dir/janus_eval_core/xvector.scp



fi

if [ $stage -le 5 ]; then

    utils/combine_data.sh data/sitw_sre18_dev_vast data/sitw_dev data/sre18_dev_vast
    mkdir -p $xvector_dir/sitw_sre18_dev_vast
    cat $xvector_dir/{sitw_dev,sre18_dev_vast}/xvector.scp > $xvector_dir/sitw_sre18_dev_vast/xvector.scp
fi

exit


exit
