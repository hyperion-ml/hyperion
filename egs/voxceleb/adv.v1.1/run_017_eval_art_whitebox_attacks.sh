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
use_trials_subset=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

if [ "$use_gpu" == "true" ];then
    eval_args="--use-gpu true"
    eval_cmd="$cuda_eval_cmd"
else
    eval_cmd="$train_cmd"
fi

if [ "$use_trials_subset" == "true" ];then
    condition=o_clean_1000_1000
else
    condition=o_clean
fi
trial_list=data/voxceleb1_test/trials_$condition

xvector_dir=exp/xvectors/$nnet_name
score_dir=exp/scores/$nnet_name

score_clean=$score_dir/cosine_cal_v1/voxceleb1_scores
cal_file=$score_dir/cosine_cal_v1/cal_tel.h5

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
	score_plda_dir=$score_dir/cosine_art_fgsm_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_fgsm_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 2 ];then
    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_fgsm_minimal_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM minimal attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.eps-step $alpha --attack.minimal" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_fgsm_minimal_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi

if [ $stage -le 3 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/cosine_art_fgml1_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGM-L1 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.norm 1" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_fgml1_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 4 ];then
    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_fgml1_minimal_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGM-L1 minimal attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.eps-step $alpha --attack.minimal --attack.norm 1" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_fgml1_minimal_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 5 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/cosine_art_fgml2_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGM-L2 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.norm 2" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_fgml2_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 6 ];then
    score_array=()
    stats_array=()

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_fgml2_minimal_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGM-L2 minimal attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type fgm --attack.eps $eps --attack.eps-step $alpha --attack.minimal --attack.norm 2" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_fgml2_minimal_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 7 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_iterfgsm_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with IterFGM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type bim --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_iterfgsm_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 8 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_pgdlinf_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD Linf attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type pgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_pgdlinf_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi


if [ $stage -le 9 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_pgdl1_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD L1 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type pgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10 --attack.norm 1" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_pgdl1_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 10 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_pgdl2_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD L2 attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type pgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10 --attack.norm 2" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_pgdl2_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 11 ];then

    for confidence in 0 #1
    do
	score_plda_dir=$score_dir/cosine_art_cwl2_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner L2 attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type cw-l2 --attack.confidence $confidence" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		$trial_list $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi


if [ $stage -le 12 ];then

    for confidence in 0 #1
    do
	score_plda_dir=$score_dir/cosine_art_cwlinf_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner Linf attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type cw-linf --attack.confidence $confidence --attack.initial-c 1e-5" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		$trial_list $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi

if [ $stage -le 14 ];then
    score_array=()
    stats_array=()
    for norm in inf 1 2 
    do
      for eps in 0.00001 0.0001 0.001 0.01 0.1
      do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_autopgdl${norm}_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with Auto-PGD $norm  attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh \
	  --cmd "$eval_cmd" $eval_args --nj 80 \
	  --feat-config $feat_config  \
	  --attack-opts "--attack.attack-type auto-pgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10 --attack.norm $norm" \
	  --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	  --cal-file $cal_file --threshold $thr005 \
	  $trial_list \
    	  data/voxceleb1_test/utt2model \
          data/voxceleb1_test \
    	  $xvector_dir/voxceleb1_test/xvector.scp \
	  $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
		   local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_autopgdl${norm}_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
				 $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
				 $score_analysis_dir/voxceleb1 &
      fi
    done
fi

if [ $stage -le 15 ];then
    score_array=()
    stats_array=()
    for norm in inf 1 2 
    do
      for eps in 0.0001 0.001 0.01 0.1
      do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_autocgdl${norm}_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with Auto-CGD $norm  attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh \
	  --cmd "$eval_cmd" $eval_args --nj 80 \
	  --feat-config $feat_config  \
	  --attack-opts "--attack.attack-type auto-cgd --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10 --attack.norm $norm" \
	  --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	  --cal-file $cal_file --threshold $thr005 \
	  $trial_list \
    	  data/voxceleb1_test/utt2model \
          data/voxceleb1_test \
    	  $xvector_dir/voxceleb1_test/xvector.scp \
	  $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
		   local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_autocgdl${norm}_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
				 $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
				 $score_analysis_dir/voxceleb1 &
      fi
    done
fi

if [ $stage -le 16 ];then
    score_array=()
    stats_array=()
    for eps in 0.0001 0.001 0.01 0.1
    do
      alpha=$(echo $eps | awk '{ print $0/5.}')
      score_plda_dir=$score_dir/cosine_art_deepfool_e${eps}
      echo "Eval Voxceleb 1 with Cosine scoring with DeepFool  attack eps=$eps"
      steps_adv/eval_cosine_scoring_from_art_test_wav.sh \
	--cmd "$eval_cmd" $eval_args --nj 80 \
	--feat-config $feat_config  \
	--attack-opts "--attack.attack-type deepfool --attack.eps $eps --attack.max-iter 100" \
	--save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	--cal-file $cal_file --threshold $thr005 \
	$trial_list \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	$nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
      
      $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
		 local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
      
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
      score_analysis_dir=$score_dir/cosine_art_deepfool_eall
      local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
			       $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
			       $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 17 ];then

    for confidence in 0 #1
    do
	score_plda_dir=$score_dir/cosine_art_elasticnet_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with ElasticNet attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type elasticnet --attack.confidence $confidence --attack.max-iter 100 --attack.lr 0.01"  \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		$trial_list $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi


if [ $stage -le 20 ];then

    for norm in inf 2
    do
	score_plda_dir=$score_dir/cosine_art_hopskipjump_norm${norm}
	echo "Eval Voxceleb 1 with Cosine scoring with Hopskipjump attack norm=$norm"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type hop-skip-jump --attack.norm $norm --attack.max-iter 50 --attack.max-eval 10000 --attack.init-eval 10 --attack.init-size 100"  \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		$trial_list $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi


if [ $stage -le 23 ];then

    for eta in 0.01
    do
	score_plda_dir=$score_dir/cosine_art_newtonfool_eta$eta
	echo "Eval Voxceleb 1 with Cosine scoring with NewtonFool attack eta=$eta"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type newtonfool --attack.eta $eta" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		$trial_list $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi

if [ $stage -le 25 ];then

    for lambda_tv in 0.3
    do
	score_plda_dir=$score_dir/cosine_art_shadow_theta$theta
	echo "Eval Voxceleb 1 with Cosine scoring with Shadow attack lambda=$lambda_tv"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type shadow --attack.lambda-tv $lambda_tv" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		$trial_list $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi

if [ $stage -le 26 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_art_wass_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with Wassertein attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type wasserstein --attack.eps $eps --attack.eps-step $alpha --attack.max-iter 10 --attack.reg 1" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
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
	score_analysis_dir=$score_dir/cosine_art_wass_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi
fi

if [ $stage -le 27 ];then

    for confidence in 0 #1
    do
	score_plda_dir=$score_dir/cosine_art_zoo_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Zoo attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
	    --feat-config $feat_config  \
	    --attack-opts "--attack.attack-type zoo --attack.confidence $confidence" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    $trial_list \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
	if [ "${do_analysis}" == "true" ];then
	    score_analysis_dir=$score_plda_dir
	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
		$trial_list $score_clean \
		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
		$score_analysis_dir/voxceleb1 &
	fi

    done

fi


# The attacks below have issues when applying to audio

# if [ $stage -le 13 ];then

#     for eps in 0.0001
#     do
#       score_plda_dir=$score_dir/cosine_art_boundary_eps${eps}
#       alpha=$(echo $eps | awk '{ print $0/5.}')
#       echo "Eval Voxceleb 1 with Cosine scoring with boundary attack eps=$eps"
#       steps_adv/eval_cosine_scoring_from_art_test_wav.sh \
# 	--cmd "$eval_cmd" $eval_args --nj 400 \
# 	--feat-config $feat_config  \
# 	--attack-opts "--attack.attack-type boundary --attack.eps $eps --attack.delta $eps --attack.max-iter 5000" \
# 	--save-wav $save_wav --save-wav-path $score_plda_dir/wav \
# 	--cal-file $cal_file --threshold $thr005 \
# 	$trial_list \
#     	data/voxceleb1_test/utt2model \
#         data/voxceleb1_test \
#     	$xvector_dir/voxceleb1_test/xvector.scp \
# 	$nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
      
#       $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
# 		 local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
      
#       for f in $(ls $score_plda_dir/*_results);
#       do
# 	echo $f
# 	cat $f
# 	echo ""
#       done
#       if [ "${do_analysis}" == "true" ];then
# 	score_analysis_dir=$score_plda_dir
# 	local/attack_analysis.sh \
# 	  --cmd "$train_cmd --mem 10G" \
# 	  $trial_list $score_clean \
# 	  $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
# 	  $score_analysis_dir/voxceleb1 &
#       fi

#     done

# fi

# it needs acces to hidden layers
# if [ $stage -le 18 ];then
#   for eps in 0.00001 0.0001 0.001 0.01 0.1
#   do
#     alpha=$(echo $eps | awk '{ print $0/5.}')
#     score_plda_dir=$score_dir/cosine_art_fadv_e${eps}
#     echo "Eval Voxceleb 1 with Cosine scoring with feature adversaries  attack eps=$eps"
#     steps_adv/eval_cosine_scoring_from_art_test_wav.sh \
#       --cmd "$eval_cmd" $eval_args --nj 80 \
#       --feat-config $feat_config  \
#       --attack-opts "--attack.attack-type feature-adv --attack.delta $eps --attack.eps-step $alpha --attack.max-iter 100" \
#       --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
#       --cal-file $cal_file --threshold $thr005 \
#       $trial_list \
#       data/voxceleb1_test/utt2model \
#       data/voxceleb1_test \
#       $xvector_dir/voxceleb1_test/xvector.scp \
#       $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    
#     $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
# 	       local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
    
#     for f in $(ls $score_plda_dir/*_results);
#     do
#       echo $f
#       cat $f
#       echo ""
#     done
    
#     score_array+=($score_plda_dir/voxceleb1_scores)
#     stats_array+=($score_plda_dir/voxceleb1_stats)
    
#   done
#   if [ "${do_analysis}" == "true" ];then
#     score_analysis_dir=$score_dir/cosine_art_fadv_eall
#     local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
# 			     $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
# 			     $score_analysis_dir/voxceleb1 &
#   fi
# fi

# if [ $stage -le 19 ];then
#     score_array=()
#     stats_array=()
#     for norm in inf 1 2 
#     do
#       for sigma in 0.0002
#       do
# 	score_plda_dir=$score_dir/cosine_art_geoda${norm}_s${sigma}
# 	echo "Eval Voxceleb 1 with Cosine scoring with GeoDA $norm sigma=$sigma"
# 	steps_adv/eval_cosine_scoring_from_art_test_wav.sh \
# 	  --cmd "$eval_cmd" $eval_args --nj 80 \
# 	  --feat-config $feat_config  \
# 	  --attack-opts "--attack.attack-type geoda --attack.max-iter 4000 --attack.sigma-geoda $sigma --attack.norm $norm" \
# 	  --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
# 	  --cal-file $cal_file --threshold $thr005 \
# 	  $trial_list \
#     	  data/voxceleb1_test/utt2model \
#           data/voxceleb1_test \
#     	  $xvector_dir/voxceleb1_test/xvector.scp \
# 	  $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
# 	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
# 		   local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
# 	for f in $(ls $score_plda_dir/*_results);
# 	do
# 	  echo $f
# 	  cat $f
# 	  echo ""
# 	done

# 	score_array+=($score_plda_dir/voxceleb1_scores)
# 	stats_array+=($score_plda_dir/voxceleb1_stats)

#       done
#       if [ "${do_analysis}" == "true" ];then
# 	score_analysis_dir=$score_dir/cosine_art_geoda${norm}_sall
# 	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
# 				 $trial_list $score_clean "${score_array[*]}" "${stats_array[*]}" \
# 				 $score_analysis_dir/voxceleb1 &
#       fi
#     done
# fi

#
# if [ $stage -le 21 ];then

#     for norm in inf 1 2
#     do
# 	score_plda_dir=$score_dir/cosine_art_brendel_norm${norm}
# 	echo "Eval Voxceleb 1 with Cosine scoring with Brendel attack norm=$norm"
# 	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
# 	    --feat-config $feat_config  \
# 	    --attack-opts "--attack.attack-type brendel --attack.norm $norm --attack.max-iter 1000 --attack.lr 1e-3 --attack.binary-search-steps 10 --attack.init-size 100"  \
# 	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
# 	    --cal-file $cal_file --threshold $thr005 \
# 	    $trial_list \
#     	    data/voxceleb1_test/utt2model \
#             data/voxceleb1_test \
#     	    $xvector_dir/voxceleb1_test/xvector.scp \
# 	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
# 	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
# 	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
# 	for f in $(ls $score_plda_dir/*_results);
# 	do
# 	    echo $f
# 	    cat $f
# 	    echo ""
# 	done
# 	if [ "${do_analysis}" == "true" ];then
# 	    score_analysis_dir=$score_plda_dir
# 	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
# 		$trial_list $score_clean \
# 		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
# 		$score_analysis_dir/voxceleb1 &
# 	fi

#     done

# fi

## it needs to train some importance vector
# if [ $stage -le 22 ];then

#     for norm in 2
#     do
# 	score_plda_dir=$score_dir/cosine_art_lowprofool_norm${norm}
# 	echo "Eval Voxceleb 1 with Cosine scoring with LowProFool attack norm=$norm"
# 	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
# 	    --feat-config $feat_config  \
# 	    --attack-opts "--attack.attack-type low-pro-fool --attack.norm $norm --attack.max-iter 100"  \
# 	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
# 	    --cal-file $cal_file --threshold $thr005 \
# 	    $trial_list \
#     	    data/voxceleb1_test/utt2model \
#             data/voxceleb1_test \
#     	    $xvector_dir/voxceleb1_test/xvector.scp \
# 	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
# 	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
# 	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
# 	for f in $(ls $score_plda_dir/*_results);
# 	do
# 	    echo $f
# 	    cat $f
# 	    echo ""
# 	done
# 	if [ "${do_analysis}" == "true" ];then
# 	    score_analysis_dir=$score_plda_dir
# 	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
# 		$trial_list $score_clean \
# 		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
# 		$score_analysis_dir/voxceleb1 &
# 	fi

#     done

# fi

## Too SLOW
# if [ $stage -le 24 ];then

#     for theta in 0.1
#     do
# 	score_plda_dir=$score_dir/cosine_art_jsma_theta$theta
# 	echo "Eval Voxceleb 1 with Cosine scoring with JSMA attack theta=$theta"
# 	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 400 \
# 	    --feat-config $feat_config  \
# 	    --attack-opts "--attack.attack-type jsma --attack.theta $theta" \
# 	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
# 	    --cal-file $cal_file --threshold $thr005 \
# 	    $trial_list \
#     	    data/voxceleb1_test/utt2model \
#             data/voxceleb1_test \
#     	    $xvector_dir/voxceleb1_test/xvector.scp \
# 	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
# 	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
# 	    local/score_voxceleb1_single_cond.sh data/voxceleb1_test $condition $score_plda_dir 
	
# 	for f in $(ls $score_plda_dir/*_results);
# 	do
# 	    echo $f
# 	    cat $f
# 	    echo ""
# 	done
# 	if [ "${do_analysis}" == "true" ];then
# 	    score_analysis_dir=$score_plda_dir
# 	    local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
# 		$trial_list $score_clean \
# 		$score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats \
# 		$score_analysis_dir/voxceleb1 &
# 	fi

#     done

# fi
