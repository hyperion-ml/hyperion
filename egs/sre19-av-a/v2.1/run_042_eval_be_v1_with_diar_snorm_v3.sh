#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

coh_vid_data=dihard2_sitw_sre18_dev_vast
coh_vast_data=dihard2_sitw_sre18_dev_vast

ncoh_vid=1000
ncoh_vast=1000

plda_label=${plda_type}y${plda_y_dim}_v1d
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda_${diar_name}


if [ ! -d data/$coh_vid_data ];then
    utils/combine_data.sh data/$coh_vid_data \
	data/dihard2_train data/sitw_sre18_dev_vast
fi
if [ ! -f $xvector_dir/${coh_vid_data}_${diar_name}/xvector.scp ];then
    mkdir -p $xvector_dir/${coh_vid_data}_${diar_name}
    cat $xvector_dir/dihard2_train/xvector.scp $xvector_dir/sitw_sre18_dev_vast_${diar_name}/xvector.scp  \
	> $xvector_dir/${coh_vid_data}_${diar_name}/xvector.scp
fi


if [ $stage -le 1 ]; then
    
    steps_be/train_vid_be_v1.sh --cmd "$train_cmd" \
	--lda_dim $lda_dim \
	--plda_type $plda_type \
	--y_dim $plda_y_dim --z_dim $plda_z_dim \
	$xvector_dir/$plda_data/xvector.scp \
	data/$plda_data \
	$xvector_dir/sitw_dev_${diar_name}/xvector.scp \
	data/sitw_dev_${diar_name} \
	$xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
	data/sre18_dev_vast_${diar_name} \
	$be_dir 
fi

score_plda_dir=$score_dir/plda_snorm_v3_ncoh${ncoh_vast}_${diar_name}

if [ $stage -le 2 ];then

    #SITW
    echo "SITW dev S-Norm with diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}
	steps_be/eval_vid_be_diar_snorm_v2.sh --cmd "$train_cmd" \
	    --plda_type $plda_type --ncoh $ncoh_vid --ncoh_discard 50 \
	    $sitw_dev_trials/$cond_i.lst \
	    data/sitw_dev_enroll/utt2spk \
	    $xvector_dir/sitw_dev_enroll/xvector.scp \
	    $xvector_dir/sitw_dev_test_${diar_name}/xvector.scp \
	    data/${coh_vid_data}/utt2spk \
	    $xvector_dir/${coh_vid_data}_${diar_name}/xvector.scp \
	    $be_dir/lda_lnorm_adapt.h5 \
	    $be_dir/plda.h5 \
	    $score_plda_dir/sitw_dev_${cond_i}_scores &
    done

    echo "SITW eval S-Norm with diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}
	steps_be/eval_vid_be_diar_snorm_v2.sh --cmd "$train_cmd" \
	    --plda_type $plda_type --ncoh $ncoh_vid \
	    $sitw_eval_trials/$cond_i.lst \
	    data/sitw_eval_enroll/utt2spk \
	    $xvector_dir/sitw_eval_enroll/xvector.scp \
	    $xvector_dir/sitw_eval_test_${diar_name}/xvector.scp \
	    data/${coh_vid_data}/utt2spk \
	    $xvector_dir/${coh_vid_data}_${diar_name}/xvector.scp \
	    $be_dir/lda_lnorm_adapt.h5 \
	    $be_dir/plda.h5 \
	    $score_plda_dir/sitw_eval_${cond_i}_scores &
    done

    wait
    local/score_sitw.sh data/sitw_dev_test dev $score_plda_dir 
    local/score_sitw.sh data/sitw_eval_test eval $score_plda_dir 
fi



if [ $stage -le 3 ]; then

    #SRE18
    echo "SRE18 S-Norm with diarization"

    
    steps_be/eval_vid_be_diar_snorm_v2.sh --cmd "$train_cmd" \
	--plda_type $plda_type --ncoh $ncoh_vast --ncoh_discard 7 \
    	$sre18_dev_trials_vast \
    	data/sre18_dev_enroll_vast/utt2spk \
	$xvector_dir/sre18_dev_enroll_vast/xvector.scp \
    	$xvector_dir/sre18_dev_test_vast_${diar_name}/xvector.scp \
    	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}_${diar_name}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre18_dev_vast_scores &

    
    steps_be/eval_vid_be_diar_snorm_v2.sh \
	--cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh_vast \
    	$sre18_eval_trials_vast \
    	data/sre18_eval_enroll_vast/utt2spk \
    	$xvector_dir/sre18_eval_enroll_vast/xvector.scp \
    	$xvector_dir/sre18_eval_test_vast_${diar_name}/xvector.scp \
    	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}_${diar_name}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre18_eval_vast_scores &

    wait

    local/score_sre18vast.sh data/sre18_dev_test_vast dev $score_plda_dir
    local/score_sre18vast.sh data/sre18_eval_test_vast eval $score_plda_dir

fi


if [ $stage -le 4 ]; then

    #SRE19
    echo "SRE19 S-Norm with diarization"
    
    steps_be/eval_vid_be_diar_snorm_v2.sh --cmd "$train_cmd" \
	--plda_type $plda_type --ncoh $ncoh_vast \
    	data/sre19_av_a_dev_test/trials \
    	data/sre19_av_a_dev_enroll/utt2spk \
    	$xvector_dir/sre19_av_a_dev_enroll/xvector.scp \
    	$xvector_dir/sre19_av_a_dev_test_${diar_name}/xvector.scp \
    	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}_${diar_name}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre19_av_a_dev_scores &


    steps_be/eval_vid_be_diar_snorm_v2.sh --cmd "$train_cmd" \
	--plda_type $plda_type --ncoh $ncoh_vast \
    	data/sre19_av_a_eval_test/trials \
    	data/sre19_av_a_eval_enroll/utt2spk \
    	$xvector_dir/sre19_av_a_eval_enroll/xvector.scp \
    	$xvector_dir/sre19_av_a_eval_test_${diar_name}/xvector.scp \
    	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}_${diar_name}/xvector.scp \
    	$be_dir/lda_lnorm_adapt2.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/sre19_av_a_eval_scores &

    wait

    local/score_sre19av.sh data/sre19_av_a_dev_test a_dev $score_plda_dir
    local/score_sre19av.sh data/sre19_av_a_eval_test a_eval $score_plda_dir

fi


if [ $stage -le 5 ]; then

    #JANUS
    echo "JANUS S-Norm with diarization"

    steps_be/eval_vid_be_diar_snorm_v2.sh --cmd "$train_cmd" \
	--plda_type $plda_type --ncoh $ncoh_vid \
    	data/janus_dev_test_core/trials \
    	data/janus_dev_enroll/utt2spk \
    	$xvector_dir/janus_dev_enroll/xvector.scp \
    	$xvector_dir/janus_dev_test_core_${diar_name}/xvector.scp \
	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}_${diar_name}/xvector.scp \
    	$be_dir/lda_lnorm_adapt.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/janus_dev_core_scores &

    
    steps_be/eval_vid_be_diar_snorm_v2.sh --cmd "$train_cmd" \
	--plda_type $plda_type --ncoh $ncoh_vid \
    	data/janus_eval_test_core/trials \
    	data/janus_eval_enroll/utt2spk \
    	$xvector_dir/janus_eval_enroll/xvector.scp \
    	$xvector_dir/janus_eval_test_core_${diar_name}/xvector.scp \
    	data/${coh_vast_data}/utt2spk \
	$xvector_dir/${coh_vast_data}_${diar_name}/xvector.scp \
    	$be_dir/lda_lnorm_adapt.h5 \
    	$be_dir/plda.h5 \
	$score_plda_dir/janus_eval_core_scores &

    wait

    local/score_janus_core.sh data/janus_dev_test_core dev $score_plda_dir
    local/score_janus_core.sh data/janus_eval_test_core eval $score_plda_dir

fi


if [ $stage -le 6 ];then
    local/calibrate_sre19av_a_v1_sre18.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_dir}_cal_v1_sre18
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_dir}_cal_v1_sre18
    local/score_sre18vast.sh data/sre18_dev_test_vast dev ${score_plda_dir}_cal_v1_sre18
    local/score_sre18vast.sh data/sre18_eval_test_vast eval ${score_plda_dir}_cal_v1_sre18
    local/score_sre19av.sh data/sre19_av_a_dev_test a_dev ${score_plda_dir}_cal_v1_sre18
    local/score_sre19av.sh data/sre19_av_a_eval_test a_eval ${score_plda_dir}_cal_v1_sre18
    local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre18
    local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre18

    local/calibrate_sre19av_a_v1_sre19.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_dir}_cal_v1_sre19
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_dir}_cal_v1_sre19
    local/score_sre18vast.sh data/sre18_dev_test_vast dev ${score_plda_dir}_cal_v1_sre19
    local/score_sre18vast.sh data/sre18_eval_test_vast eval ${score_plda_dir}_cal_v1_sre19
    local/score_sre19av.sh data/sre19_av_a_dev_test a_dev ${score_plda_dir}_cal_v1_sre19
    local/score_sre19av.sh data/sre19_av_a_eval_test a_eval ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre19
    exit
fi


exit

