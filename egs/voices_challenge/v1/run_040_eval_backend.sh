#!/bin/bash
# Copyright       2019   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
ncoh=3000
ncoh_discard=400

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda
voices_scorer=$voices_root/Development_Data/Speaker_Recognition/voices_scorer


if [ $stage -le 1 ]; then
  echo "Train PLDA Back-end"
  steps_be/train_be_v1.sh --cmd "$train_cmd" \
    --lda_dim $lda_dim \
    --plda_type $plda_type \
    --y_dim $plda_y_dim --z_dim $plda_z_dim \
    $xvector_dir/$plda_data/xvector.scp \
    data/$plda_data \
    $xvector_dir/voices19_challenge_dev/xvector.scp \
    data/voices19_challenge_dev \
    $be_dir 
fi


if [ $stage -le 2 ];then

  echo "Voices19 dev"
  steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
    data/voices19_challenge_dev_test/trials \
    data/voices19_challenge_dev_enroll/utt2model \
    $xvector_dir/voices19_challenge_dev/xvector.scp \
    $be_dir/lda_lnorm_adapt.h5 \
    $be_dir/plda.h5 \
    $score_plda_dir/voices19_challenge_dev_scores &


  echo "Voices19 eval"
  steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
    data/voices19_challenge_eval_test/trials \
    data/voices19_challenge_eval_enroll/utt2model \
    $xvector_dir/voices19_challenge_eval/xvector.scp \
    $be_dir/lda_lnorm_adapt.h5 \
    $be_dir/plda.h5 \
    $score_plda_dir/voices19_challenge_eval_scores &
  wait

  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test dev $score_plda_dir
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval $score_plda_dir

fi

if [ $stage -le 3 ];then
  local/calibrate_voices19_challenge_v1.sh --cmd "$train_cmd" $score_plda_dir
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test dev ${score_plda_dir}_cal_v1
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval ${score_plda_dir}_cal_v1
fi



score_plda_dir=$score_dir/plda_snorm


if [ $stage -le 4 ];then

  echo "Voices19 dev S-Norm"
  steps_be/eval_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh --ncoh_discard $ncoh_discard \
    data/voices19_challenge_dev_test/trials \
    data/voices19_challenge_dev_enroll/utt2model \
    $xvector_dir/voices19_challenge_dev/xvector.scp \
    data/${coh_data}/utt2spk \
    $xvector_dir/${coh_data}/xvector.scp \
    $be_dir/lda_lnorm_adapt.h5 \
    $be_dir/plda.h5 \
    $score_plda_dir/voices19_challenge_dev_scores &

  echo "Voices19 eval S-Norm"
  steps_be/eval_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
    data/voices19_challenge_eval_test/trials \
    data/voices19_challenge_eval_enroll/utt2model \
    $xvector_dir/voices19_challenge_eval/xvector.scp \
    data/${coh_data}/utt2spk \
    $xvector_dir/${coh_data}/xvector.scp \
    $be_dir/lda_lnorm_adapt.h5 \
    $be_dir/plda.h5 \
    $score_plda_dir/voices19_challenge_eval_scores &

  wait
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test dev $score_plda_dir
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval $score_plda_dir
fi

if [ $stage -le 5 ];then
  local/calibrate_voices19_challenge_v1.sh --cmd "$train_cmd" $score_plda_dir
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test dev ${score_plda_dir}_cal_v1
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval ${score_plda_dir}_cal_v1
fi
