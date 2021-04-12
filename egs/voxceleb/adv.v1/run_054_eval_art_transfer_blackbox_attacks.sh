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
do_analysis=false
save_wav=false
feat_config=conf/fbank80_stmn_16k.yaml

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

transfer_feat_config=$feat_config

if [ "$use_gpu" == "true" ];then
    eval_args="--use-gpu true"
    eval_cmd="$cuda_eval_cmd"
else
    eval_cmd="$train_cmd"
fi

xvector_dir=exp/xvectors/$nnet_name
score_dir=exp/scores/$nnet_name

score_clean=$score_dir/cosine_cal_v1/voxceleb1_scores
cal_file=$score_dir/cosine_cal_v1/cal_tel.h5

transfer_xvector_dir=exp/xvectors/$transfer_nnet_name
transfer_score_dir=exp/scores/$transfer_nnet_name
transfer_cal_file=$transfer_score_dir/cosine_cal_v1/cal_tel.h5

#thresholds for p=(0.05,0.01,0.001) -> thr=(2.94, 4.60, 6.90)
thr005=2.94
thr001=4.60
thr0001=6.90
declare -a score_array
declare -a stats_array

if [ $stage -le 1 ];then

    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgsm_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgsm_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 2 ];then

    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgsm_minimal_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.minimal" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgsm_minimal_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi



if [ $stage -le 3 ];then

    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml1_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGM L1 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.norm 1" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml1_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 4 ];then

    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml1_minimal_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGM minimal L1 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.minimal --attack.norm 1" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml1_minimal_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi



if [ $stage -le 5 ];then

    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml2_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGM L2 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.norm 2" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml2_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 6 ];then

    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml2_minimal_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring FGM minimal L2 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.minimal --attack.norm 2" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_fgml2_minimal_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 7 ];then
    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_iterfgsm_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with iter FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type bim --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_iterfgsm_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi



if [ $stage -le 8 ];then
    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_pgdlinf_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD Linf attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type pgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_pgdlinf_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 9 ];then
    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_pgdl1_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD L1 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type pgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10 --attack.norm 1" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_pgdl1_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 10 ];then
    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_pgdl2_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD L2 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type pgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10 --attack.norm 2" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_pgdl2_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 11 ];then

    for confidence in 0 #1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_cwl2_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner L2 attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type cw-l2 --attack.confidence $confidence" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		data/voxceleb1_test/trials_o_clean $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi


if [ $stage -le 12 ];then

    for confidence in 0 #1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/transfer.${transfer_nnet_name}/cosine_art_cwlinf_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner LInf attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_transfer_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 40 \
	    --feat-config $feat_config  \
	    --transfer-feat-config $transfer_feat_config  \
	    --attack-opts "--attack.attack-type cw-linf --attack.confidence $confidence --attack.eps 0.3" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --transfer-cal-file $transfer_cal_file \
	    --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet \
    	    $transfer_xvector_dir/voxceleb1_test/xvector.scp \
	    $transfer_nnet \
	    $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		data/voxceleb1_test/trials_o_clean $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi

wait

