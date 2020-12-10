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
	score_plda_dir=$score_dir/cosine_fgsm_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config $feat_config  \
	    --attack-type fgsm --eps $eps \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done
    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/cosine_fgsm_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 2 ];then
    score_array=()
    stats_array=()
    for snr in 30 20 10 0
    do
	score_plda_dir=$score_dir/cosine_fgsm_snr${snr}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM attack snr=$snr"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config $feat_config  \
	    --attack-type snr-fgsm --snr $snr \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/cosine_fgsm_snrall
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
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_randfgsm_e${eps}_a${alpha}
	echo "Eval Voxceleb 1 with Cosine scoring with Rand-FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config $feat_config  \
	    --attack-type rand-fgsm --eps $eps --alpha $alpha \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/cosine_randfgsm_eall
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
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_iterfgsm_e${eps}_a${alpha}
	echo "Eval Voxceleb 1 with Cosine scoring with Iterative FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config $feat_config  \
	    --attack-type iter-fgsm --eps $eps --alpha $alpha \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/cosine_iterfgsm_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi


if [ $stage -le 5 ];then

    for confidence in 0 #1
    do
	for lr in 0.001
	do
	    for it in 10
	    do
		
		score_plda_dir=$score_dir/cosine_cwl2_conf${confidence}_lr${lr}_noabort_it$it
		echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner L2 attack confidence=$confidence lr=$lr num-its=$it"
		steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 100 \
		    --feat-config $feat_config  \
		    --attack-type cw-l2 --confidence $confidence \
		    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
		    --cal-file $cal_file --threshold $thr005 --lr $lr --attack-opt "--attack-no-abort" --max-iter $it \
		    data/voxceleb1_test/trials_o_clean \
    		    data/voxceleb1_test/utt2model \
		    data/voxceleb1_test \
    		    $xvector_dir/voxceleb1_test/xvector.scp \
		    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    		
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
	done
    done
    
fi


if [ $stage -le 6 ];then

    for confidence in 0 #1
    do
	for lr in 0.001 
	do
	    for it in 10
	    do
		score_plda_dir=$score_dir/cosine_cwrms_conf${confidence}_lr${lr}_noabort_it$it
		echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner RMS attack confidence=$confidence lr=$lr num_its=$it"
		steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd -tc 15" $eval_args --nj 100 \
		    --feat-config $feat_config  \
		    --attack-type cw-l2 --confidence $confidence \
		    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
		    --cal-file $cal_file --threshold $thr005 --lr $lr --attack-opt "--attack-no-abort --attack-norm-time" \
		    --max-iter $it \
		    data/voxceleb1_test/trials_o_clean \
    		    data/voxceleb1_test/utt2model \
		    data/voxceleb1_test \
    		    $xvector_dir/voxceleb1_test/xvector.scp \
		    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    		
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
	done
    done

fi
exit

if [ $stage -le 7 ];then

    for confidence in 0 #1
    do
	for lr in 0.001 
	do
	    for it in 10
	    do
		score_plda_dir=$score_dir/cosine_cwsnr_conf${confidence}_lr${lr}_noabort_it$it
		echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner SNR attack confidence=$confidence lr=$lr num_its=$it"
		steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd -tc 15" $eval_args --nj 100 \
		    --feat-config $feat_config  \
		    --attack-type cw-l2 --confidence $confidence \
		    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
		    --cal-file $cal_file --threshold $thr005 --lr $lr --attack-opt "--attack-no-abort --attack-use-snr" \
		    --max-iter $it \
		    data/voxceleb1_test/trials_o_clean \
    		    data/voxceleb1_test/utt2model \
		    data/voxceleb1_test \
    		    $xvector_dir/voxceleb1_test/xvector.scp \
		    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    		
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
	done
    done
fi



if [ $stage -le -8 ];then

    for confidence in 0 1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_cwl0_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner L0 attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 1000 \
	    --feat-config $feat_config  \
	    --attack-type cw-l0 --confidence $confidence --c-factor 10 \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
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



if [ $stage -le 9 ];then

    for confidence in 0 1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_cwlinf_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner LInf attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 40 \
	    --feat-config $feat_config  \
	    --attack-type cw-linf --confidence $confidence --c-factor 2 \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats
    	
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



if [ $stage -le 10 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_pgdlinf_e${eps}_a${alpha}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD Linf attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config $feat_config  \
	    --attack-type pgd --eps $eps --alpha $alpha --attack-opt "" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/cosine_pgdlinf_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi

if [ $stage -le 11 ];then
    score_array=()
    stats_array=()
    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_pgdlinf_randinit5_e${eps}_a${alpha}
	echo "Eval Voxceleb 1 with Cosine scoring with PGD Linf attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_adv_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 100 \
	    --feat-config $feat_config  \
	    --attack-type pgd --eps $eps --alpha $alpha --attack-opt "--attack-num-random-init 5" \
	    --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
	    --cal-file $cal_file --threshold $thr005 \
	    data/voxceleb1_test/trials_o_clean \
    	    data/voxceleb1_test/utt2model \
            data/voxceleb1_test \
    	    $xvector_dir/voxceleb1_test/xvector.scp \
	    $nnet $score_plda_dir/voxceleb1_scores $score_plda_dir/voxceleb1_stats

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)
    	
	$train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_o_clean.sh data/voxceleb1_test $score_plda_dir 
	
	for f in $(ls $score_plda_dir/*_results);
	do
	    echo $f
	    cat $f
	    echo ""
	done

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/cosine_pgdlinf_randinit_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi




wait
