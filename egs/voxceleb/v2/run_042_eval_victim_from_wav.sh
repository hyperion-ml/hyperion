#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
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
. datapath.sh 

if [ "$use_gpu" == "true" ];then
    eval_args="--use-gpu true"
    eval_cmd="$cuda_eval_cmd"
else
    eval_cmd="$train_cmd"
fi

xvector_dir=exp/xvectors/$nnet_name
score_dir=exp/scores/$nnet_name
cal_file=$score_dir/cosine_cal_v1/cal_tel.h5
score_plda_dir=$score_dir/cosine_from_wav

if [ $stage -le 1 ];then

    echo "Eval Voxceleb 1 with Cosine scoring"
    steps_xvec/eval_cosine_scoring_from_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 200 \
	--feat-config conf/fbank80_stmn_16k.yaml \
	--cal-file $cal_file \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	$nnet $score_plda_dir/voxceleb1_scores
    	
    $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 

    for f in $(ls $score_plda_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi


