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

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 


nnet_name=$transfer_nnet_name
xvector_dir=exp/xvectors/$nnet_name
score_dir=exp/scores/$nnet_name
score_plda_dir=$score_dir/cosine

if [ $stage -le 1 ];then

    echo "Eval Voxceleb 1 with Cosine scoring"
    steps_be/eval_be_cos.sh --cmd "$train_cmd" \
    	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
    	$xvector_dir/voxceleb1_test/xvector.scp \
    	$score_plda_dir/voxceleb1_scores

    $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 

    for f in $(ls $score_plda_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi

if [ $stage -le 2 ];then
    local/calibrate_voxceleb1_o_clean.sh --cmd "$train_cmd" $score_plda_dir

    $train_cmd --mem 10G ${score_plda_dir}_cal_v1/log/score_voxceleb1.log \
	local/score_voxceleb1_o_clean.sh data/voxceleb1_test ${score_plda_dir}_cal_v1
    $train_cmd --mem 10G ${score_plda_dir}_cal_v1/log/score_voxceleb1_subset.log \
	local/score_voxceleb1_single_cond.sh data/voxceleb1_test o_clean_1000_1000 ${score_plda_dir}_cal_v1

    for f in $(ls ${score_plda_dir}_cal_v1/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi
