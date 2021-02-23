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
    for name in sitw_dev_test sitw_eval_test \
	sre18_eval_test_vast sre18_dev_test_vast \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_test_core janus_eval_test_core
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

if [ $stage -le 2 ]; then
    # combine datasets    
    mkdir -p $xvector_dir/sitw_dev_${diar_name}
    cat $xvector_dir/sitw_dev_{enroll,test_${diar_name}}/xvector.scp \
	> $xvector_dir/sitw_dev_${diar_name}/xvector.scp

    # mkdir -p $xvector_dir/sitw_eval_${diar_name}
    # cat $xvector_dir/sitw_eval_{enroll,test_${diar_name}}/xvector.scp \
    # 	> $xvector_dir/sitw_eval_${diar_name}/xvector.scp
    
    mkdir -p $xvector_dir/sre18_dev_vast_${diar_name}
    cat $xvector_dir/sre18_dev_{enroll_vast,test_vast_${diar_name}}/xvector.scp \
	> $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp

    # mkdir -p $xvector_dir/sre18_eval_vast_${diar_name}
    # cat $xvector_dir/sre18_eval_{enroll_vast,test_vast_${diar_name}}/xvector.scp \
    # 	> $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp

    # mkdir -p $xvector_dir/sre19_av_a_dev_${diar_name}
    # cat $xvector_dir/sre19_av_a_dev_{enroll,test_${diar_name}}/xvector.scp \
    # 	> $xvector_dir/sre19_av_a_dev_${diar_name}/xvector.scp

    # mkdir -p $xvector_dir/sre19_av_a_eval_${diar_name}
    # cat $xvector_dir/sre19_av_a_eval_{enroll,test_${diar_name}}/xvector.scp \
    # 	> $xvector_dir/sre19_av_a_eval_${diar_name}/xvector.scp

    # mkdir -p $xvector_dir/janus_dev_core_${diar_name}
    # cat $xvector_dir/janus_dev_{enroll,test_core_${diar_name}}/xvector.scp > \
    # 	$xvector_dir/janus_dev_core_${diar_name}/xvector.scp

    # mkdir -p $xvector_dir/janus_eval_core_${diar_name}
    # cat $xvector_dir/janus_eval_{enroll,test_core_${diar_name}}/xvector.scp \
    # 	> $xvector_dir/janus_eval_core_${diar_name}/xvector.scp

fi

if [ $stage -le 3 ]; then
    mkdir -p $xvector_dir/sitw_sre18_dev_vast_${diar_name}
    cat $xvector_dir/{sitw_dev_${diar_name},sre18_dev_vast_${diar_name}}/xvector.scp \
	> $xvector_dir/sitw_sre18_dev_vast_${diar_name}/xvector.scp
fi

exit
