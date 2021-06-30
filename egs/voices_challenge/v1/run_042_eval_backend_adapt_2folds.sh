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

w_mu=1
w_b=0
w_w=0.35

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

plda_label=${plda_type}y${plda_y_dim}_v2
be_name=lda${lda_dim}_${plda_label}_${plda_data}_adapt_w${w_w}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda
voices_scorer=$voices_root/Development_Data/Speaker_Recognition/voices_scorer


if [ $stage -le 2 ]; then
  echo "Trains back-end for each fold"
  echo "back-end to eval fold 1 does centering with fold2 and viceversa"
  
  steps_be/train_be_v2.sh --cmd "$train_cmd" \
    --lda_dim $lda_dim \
    --plda_type $plda_type \
    --y_dim $plda_y_dim --z_dim $plda_z_dim \
    --w_mu $w_mu --w_b $w_b --w_w $w_w \
    $xvector_dir/$plda_data/xvector.scp \
    data/$plda_data \
    $xvector_dir/voices19_challenge_dev/xvector.scp \
    data/voices19_challenge_dev_f2 \
    ${be_dir}_f1 &

  steps_be/train_be_v2.sh --cmd "$train_cmd" \
    --lda_dim $lda_dim \
    --plda_type $plda_type \
    --y_dim $plda_y_dim --z_dim $plda_z_dim \
    --w_mu $w_mu --w_b $w_b --w_w $w_w \
    $xvector_dir/$plda_data/xvector.scp \
    data/$plda_data \
    $xvector_dir/voices19_challenge_dev/xvector.scp \
    data/voices19_challenge_dev_f1 \
    ${be_dir}_f2 &

  echo "Trains back-end adapted to both folds, used for eval"
  steps_be/train_be_v2.sh --cmd "$train_cmd" \
    --lda_dim $lda_dim \
    --plda_type $plda_type \
    --y_dim $plda_y_dim --z_dim $plda_z_dim \
    --w_mu $w_mu --w_b $w_b --w_w $w_w \
    $xvector_dir/$plda_data/xvector.scp \
    data/$plda_data \
    $xvector_dir/voices19_challenge_dev/xvector.scp \
    data/voices19_challenge_dev \
    ${be_dir} &

  wait

fi


if [ $stage -le 3 ];then

  echo "Voices19 dev"
  for fold in 1 2
  do
    steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
      data/voices19_challenge_dev_test_f${fold}/trials \
      data/voices19_challenge_dev_enroll_f${fold}/utt2model \
      $xvector_dir/voices19_challenge_dev/xvector.scp \
      ${be_dir}_f${fold}/lda_lnorm_adapt.h5 \
      ${be_dir}_f${fold}/plda_adapt.h5 \
      ${score_plda_dir}_f${fold}/voices19_challenge_dev_scores &
  done

  echo "Voices19 eval"
  steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
    data/voices19_challenge_eval_test/trials \
    data/voices19_challenge_eval_enroll/utt2model \
    $xvector_dir/voices19_challenge_eval/xvector.scp \
    ${be_dir}/lda_lnorm_adapt.h5 \
    ${be_dir}/plda_adapt.h5 \
    ${score_plda_dir}/voices19_challenge_eval_scores &
  
  wait
  
  echo  "merge scores from both folds"
  mkdir -p ${score_plda_dir}_2folds
  cat ${score_plda_dir}_f{1,2}/voices19_challenge_dev_scores > ${score_plda_dir}_2folds/voices19_challenge_dev_scores

  echo "evaluate performance on merged scores"
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test_2folds dev ${score_plda_dir}_2folds
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval ${score_plda_dir}
fi

if [ $stage -le 4 ];then
  echo "Calibrate scores"
  local/calibrate_voices19_challenge_2folds_v1.sh --cmd "$train_cmd" $score_plda_dir
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test_2folds dev ${score_plda_dir}_2folds_cal_v1
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval ${score_plda_dir}_2folds_cal_v1
fi


score_plda_dir=$score_dir/plda_snorm

if [ $stage -le 5 ];then


  echo "Voices19 dev S-Norm"
  for fold in 1 2
  do
    if [ $fold -eq 1 ];then coh_fold=2; else coh_fold=1; fi

    steps_be/eval_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
      data/voices19_challenge_dev_test_f${fold}/trials \
      data/voices19_challenge_dev_enroll_f${fold}/utt2model \
      $xvector_dir/voices19_challenge_dev/xvector.scp \
      data/${coh_data}_f${coh_fold}/utt2spk \
      $xvector_dir/${coh_data}/xvector.scp \
      ${be_dir}_f${fold}/lda_lnorm_adapt.h5 \
      ${be_dir}_f${fold}/plda_adapt.h5 \
      ${score_plda_dir}_f${fold}/voices19_challenge_dev_scores &
  done
  
  echo "Voices19 eval S-Norm"
  steps_be/eval_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
    data/voices19_challenge_eval_test/trials \
    data/voices19_challenge_eval_enroll/utt2model \
    $xvector_dir/voices19_challenge_eval/xvector.scp \
    data/${coh_data}/utt2spk \
    $xvector_dir/${coh_data}/xvector.scp \
    ${be_dir}/lda_lnorm_adapt.h5 \
    ${be_dir}/plda_adapt.h5 \
    ${score_plda_dir}/voices19_challenge_eval_scores &

  wait
  
  echo "merge scores from both folds"
  mkdir -p ${score_plda_dir}_2folds
  cat ${score_plda_dir}_f{1,2}/voices19_challenge_dev_scores > ${score_plda_dir}_2folds/voices19_challenge_dev_scores

  echo "evaluate performance on merged scores"
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test_2folds dev ${score_plda_dir}_2folds
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval ${score_plda_dir}

fi

if [ $stage -le 6 ];then
  echo "Calibrate scores"
  local/calibrate_voices19_challenge_2folds_v1.sh --cmd "$train_cmd" $score_plda_dir
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_dev_test_2folds dev ${score_plda_dir}_2folds_cal_v1
  local/score_voices19_challenge.sh $voices_scorer data/voices19_challenge_eval_test eval ${score_plda_dir}_2folds_cal_v1
fi


