#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
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

# SITW Trials
sitw_dev_trials_diar=data/sitw_dev_test_${diar_name}/trials
sitw_eval_trials_diar=data/sitw_eval_test_${diar_name}/trials

# SRE18 trials
sre18_dev_trials_vast_diar=data/sre18_dev_test_vast_${diar_name}/trials
sre18_eval_trials_vast_diar=data/sre18_eval_test_vast_${diar_name}/trials

xvector_dir=exp/xvectors/$nnet_name
be_tel_dir=exp/be/$nnet_name/$be_tel_name
be_vid_dir=exp/be/$nnet_name/$be_vid_name

score_dir=exp/scores/$nnet_name/tel_${be_tel_name}_vid_${be_vid_name}
score_plda_dir=$score_dir/plda_snorm
score_plda_diar_dir=$score_dir/plda_${diar_name}_snorm
score_sre18_dir=$score_dir/sre18_plda_${diar_name}_snorm


if [ $stage -le 1 ];then

    #SITW
    echo "SITW dev S-Norm with diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}

	steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $ncoh_vid --ncoh_discard 50 \
					      $sitw_dev_trials/$cond_i.lst $sitw_dev_trials_diar/$cond_i.lst \
					      data/sitw_dev_enroll/utt2spk \
					      $xvector_dir/sitw_dev_${diar_name}/xvector.scp \
					      data/sitw_dev_test_${diar_name}/utt2orig \
					      data/${coh_vid_data}/utt2spk \
					      $xvector_dir/${coh_vid_data}/xvector.scp \
					      $be_vid_dir/lda_lnorm_adapt.h5 \
					      $be_vid_dir/plda.h5 \
					      ${score_plda_diar_dir}/sitw_dev_${cond_i}_scores &
    done

    
    echo "SITW eval S-Norm with diarization"
    for((i=0; i<${#sitw_conds[@]};i++))
    do
	cond_i=${sitw_conds[$i]}

	steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $ncoh_vid \
					      $sitw_eval_trials/$cond_i.lst $sitw_eval_trials_diar/$cond_i.lst \
					      data/sitw_eval_enroll/utt2spk \
					      $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
					      data/sitw_eval_test_${diar_name}/utt2orig \
					      data/${coh_vid_data}/utt2spk \
					      $xvector_dir/${coh_vid_data}/xvector.scp \
					      $be_vid_dir/lda_lnorm_adapt.h5 \
					      $be_vid_dir/plda.h5 \
					      ${score_plda_diar_dir}/sitw_eval_${cond_i}_scores &
    done
    wait

    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_diar_dir}
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_diar_dir}
    
fi


if [ $stage -le 2 ]; then

    #SRE18
    echo "SRE18 S-Norm with diarization"
    
    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $ncoh_vast --ncoh_discard 7 \
    					  $sre18_dev_trials_vast $sre18_dev_trials_vast_diar \
    					  data/sre18_dev_enroll_vast/utt2spk \
    					  $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
					  data/sre18_dev_test_vast_${diar_name}/utt2orig \
					  data/${coh_vast_data}/utt2spk \
					  $xvector_dir/${coh_vast_data}/xvector.scp \
    					  $be_vid_dir/lda_lnorm_adapt2.h5 \
    					  $be_vid_dir/plda.h5 \
    					  ${score_plda_diar_dir}/sre18_dev_vast_scores &

    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $ncoh_vast \
    					  $sre18_eval_trials_vast $sre18_eval_trials_vast_diar \
    					  data/sre18_eval_enroll_vast/utt2spk \
    					  $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp \
					  data/sre18_eval_test_vast_${diar_name}/utt2orig \
					  data/${coh_vast_data}/utt2spk \
					  $xvector_dir/${coh_vast_data}/xvector.scp \
    					  $be_vid_dir/lda_lnorm_adapt2.h5 \
    					  $be_vid_dir/plda.h5 \
    					  ${score_plda_diar_dir}/sre18_eval_vast_scores &
    wait
    
    local/score_sre18.sh $sre18_dev_root dev $score_plda_dir/sre18_dev_cmn2_scores ${score_plda_diar_dir}/sre18_dev_vast_scores ${score_sre18_dir}
    local/score_sre18.sh $sre18_eval_root eval $score_plda_dir/sre18_eval_cmn2_scores ${score_plda_diar_dir}/sre18_eval_vast_scores ${score_sre18_dir}

fi

if [ $stage -le 3 ];then
    local/calibrate_sitw_v1.sh --cmd "$train_cmd" $score_plda_diar_dir
    local/calibrate_sre18_v1.sh --cmd "$train_cmd" $score_plda_dir $score_plda_diar_dir
    local/score_sitw.sh data/sitw_dev_test dev ${score_plda_diar_dir}_cal_v1
    local/score_sitw.sh data/sitw_eval_test eval ${score_plda_diar_dir}_cal_v1
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_dir}_cal_v1/sre18_dev_cmn2_scores ${score_plda_diar_dir}_cal_v1/sre18_dev_vast_scores ${score_sre18_dir}_cal_v1
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_dir}_cal_v1/sre18_eval_cmn2_scores ${score_plda_diar_dir}_cal_v1/sre18_eval_vast_scores ${score_sre18_dir}_cal_v1
    
fi

    
exit


