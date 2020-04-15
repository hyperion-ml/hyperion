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

plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
score_dir=exp/scores/$nnet_name

cal_file=$score_dir/cosine_cal_v1/cal_tel.h5

#thresholds for p=(0.05,0.01,0.001) -> thr=(2.94, 4.60, 6.90)
thr005=2.94
thr001=4.60
thr0001=6.90

declare -a score_array
declare -a stats_array

if [ $stage -le 1 ];then

    for eps in 0.00001 #0.0001 0.001 0.01 0.1
    do
	score_plda_dir=$score_dir/cosine_art_fgsm_e${eps}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 80 \
	    --feat-config conf/fbank80_16k.pyconf --audio-feat logfb \
	    --attack-type fgm --attack-opt "--attack-eps $eps" \
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

	score_array+=($score_plda_dir/voxceleb1_scores)
	stats_array+=($score_plda_dir/voxceleb1_stats)

    done
    if [ "${do_analysis}" == "true" ];then
	score_analysis_dir=$score_dir/cosine_art_fgsm_eall
	local/attack_analysis.sh --cmd "$train_cmd --mem 10G" \
	    data/voxceleb1_test/trials_o_clean $score_clean "${score_array[*]}" "${stats_array[*]}" \
	    $score_analysis_dir/voxceleb1 &
    fi

fi
exit

if [ $stage -le 2 ];then

    for snr in 30 20 10 0
    do
	score_plda_dir=$score_dir/cosine_fgsm_snr${snr}
	echo "Eval Voxceleb 1 with Cosine scoring with FGSM attack snr=$snr"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config conf/fbank80_16k.pyconf --audio-feat logfb \
	    --attack-type snr-fgsm --snr $snr --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
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
    done

fi


if [ $stage -le 3 ];then

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_randfgsm_e${eps}_a${alpha}
	echo "Eval Voxceleb 1 with Cosine scoring with Rand-FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config conf/fbank80_16k.pyconf --audio-feat logfb \
	    --attack-type rand-fgsm --eps $eps --alpha $alpha --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
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
    done

fi


if [ $stage -le 4 ];then

    for eps in 0.00001 0.0001 0.001 0.01 0.1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_iterfgsm_e${eps}_a${alpha}
	echo "Eval Voxceleb 1 with Cosine scoring with Iterative FGSM attack eps=$eps"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config conf/fbank80_16k.pyconf --audio-feat logfb \
	    --attack-type iter-fgsm --eps $eps --alpha $alpha --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
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
    done

fi


if [ $stage -le 5 ];then

    for confidence in 0 1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_cwl2_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner L2 attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 20 \
	    --feat-config conf/fbank80_16k.pyconf --audio-feat logfb \
	    --attack-type cw-l2 --confidence $confidence --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
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
    done

fi


if [ $stage -le -6 ];then

    for confidence in 0 1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_cwl0_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner L0 attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 1000 \
	    --feat-config conf/fbank80_16k.pyconf --audio-feat logfb \
	    --attack-type cw-l0 --confidence $confidence --c-factor 10 --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
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
    done

fi



if [ $stage -le 7 ];then

    for confidence in 0 1
    do
	alpha=$(echo $eps | awk '{ print $0/5.}')
	score_plda_dir=$score_dir/cosine_cwlinf_conf${confidence}
	echo "Eval Voxceleb 1 with Cosine scoring with Carlini-Wagner LInf attack confidence=$confidence"
	steps_adv/eval_cosine_scoring_from_art_test_wav.sh --cmd "$eval_cmd" $eval_args --nj 40 \
	    --feat-config conf/fbank80_16k.pyconf --audio-feat logfb \
	    --attack-type cw-linf --confidence $confidence --c-factor 2 --save-wav $save_wav --save-wav-path $score_plda_dir/wav \
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
    done

fi


