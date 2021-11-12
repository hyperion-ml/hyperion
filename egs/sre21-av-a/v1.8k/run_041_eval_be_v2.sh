#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
# This scripts runs an adapted Centering+PCA+LN+PLDA back-end
# Centering is adapted per source/language
# Total cov for PLDA is average of covariance of each condition
# Single PLDA adapted from full vox+sre to CMN/YUE speakers
#

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

ncoh=5000
coh_data=voxceleb_sre_alllangs_mixfs
ft=1
pca_var_r=0.5
lda_dim=200
plda_y_dim=150
plda_z_dim=200
r_mu=100000
r_s=100000
w_mu=0.5
w_B=0.75
w_W=0.5

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
fi

pca_label=pca${pca_var_r}_rmu${r_mu}_rs${r_s}
plda_label=${plda_type}y${plda_y_dim}_adapt_wmu${w_mu}_wb${w_B}_ww${w_W}
be_name=${pca_label}_${plda_label}_v2

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda

if [ $stage -le 0 ];then
  for name in voxcelebcat sre21_audio_eval_enroll sre21_audio_eval_test sre21_audio-visual_eval_test
  do
    cp ../v1.16k/data/$name/utt2est_lang data/$name
  done
fi

if [ $stage -le 1 ]; then
  echo "Train PLDA V2"
  steps_be/train_be_plda_v2.sh \
    --cmd "$train_cmd" \
    --pca-var-r $pca_var_r --r-mu $r_mu --r-s $r_s \
    --plda_type $plda_type \
    --y_dim $plda_y_dim --z_dim $plda_z_dim \
    --w-mu $w_mu --w-B $w_B --w-W $w_W \
    $xvector_dir \
    $be_dir

fi

if [ ! -f data/sre16_eval40_yue_enroll/utt2lang ];then
  awk '{ print $1,"YUE" }' data/sre16_eval40_yue_enroll/utt2spk \
      > data/sre16_eval40_yue_enroll/utt2lang
fi

if [ ! -f data/sre16_eval40_yue_test/utt2lang ];then
    awk '{ print $1,"YUE" }' data/sre16_eval40_yue_test/utt2spk \
	> data/sre16_eval40_yue_test/utt2lang
fi

if [ $stage -le 2 ]; then
    #SRE superset and 16
    echo "SRE Superset Dev"
    steps_be/eval_be_plda_v2_cts.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      data/sre_cts_superset_8k_dev/trials \
      data/sre_cts_superset_8k_dev/utt2enroll \
      $xvector_dir/sre_cts_superset_8k_dev/xvector.scp \
      data/sre_cts_superset_8k_dev/utt2lang \
      data/sre_cts_superset_8k_dev/utt2lang \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre_cts_superset_dev_scores &

    echo "SRE16"
    steps_be/eval_be_plda_v2_cts.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      data/sre16_eval40_yue_test/trials \
      data/sre16_eval40_yue_enroll/utt2spk \
      $xvector_dir/sre16_eval40_yue/xvector.scp \
      data/sre16_eval40_yue_enroll/utt2lang \
      data/sre16_eval40_yue_test/utt2lang \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre16_eval40_yue_scores &
    
    wait

    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre_cts_superset.sh data/sre_cts_superset_8k_dev $score_plda_dir
fi


if [ $stage -le 3 ]; then

    #SRE21
    echo "SRE21 Audio Dev"
    steps_be/eval_be_plda_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio_dev/xvector.scp \
      data/sre21_audio_dev_enroll/utt2lang \
      data/sre21_audio_dev_test/utt2lang \
      data/sre21_audio_dev_enroll/segments.csv \
      data/sre21_audio_dev_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio_dev_scores &

    echo "SRE21 Audio-Visual Dev"
    steps_be/eval_be_plda_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio-visual_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_dev/xvector.scp \
      data/sre21_audio_dev_enroll/utt2lang \
      data/sre21_audio-visual_dev_test/utt2lang \
      data/sre21_audio_dev_enroll/segments.csv \
      data/sre21_audio-visual_dev_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio-visual_dev_scores &

    echo "SRE21 Audio Eval"
    steps_be/eval_be_plda_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio_eval/xvector.scp \
      data/sre21_audio_eval_enroll/utt2est_lang \
      data/sre21_audio_eval_test/utt2est_lang \
      data/sre21_audio_eval_enroll/segments.csv \
      data/sre21_audio_eval_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio_eval_scores &

    echo "SRE21 Audio-Visual Eval"
    steps_be/eval_be_plda_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio-visual_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_eval/xvector.scp \
      data/sre21_audio_eval_enroll/utt2est_lang \
      data/sre21_audio-visual_eval_test/utt2est_lang \
      data/sre21_audio_eval_enroll/segments.csv \
      data/sre21_audio-visual_eval_test/segments.csv \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio-visual_eval_scores &

    wait

    local/score_sre21.sh data/sre21_audio_dev_test audio_dev $score_plda_dir
    local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev $score_plda_dir
    local/score_sre21.sh data/sre21_audio_eval_test audio_eval $score_plda_dir
    local/score_sre21.sh data/sre21_audio-visual_eval_test audio-visual_eval $score_plda_dir
    local/score_sre21_official.sh $sre21_dev_root audio dev $score_plda_dir
    local/score_sre21_official.sh $sre21_eval_root audio eval $score_plda_dir
fi

if [ $stage -le 4 ];then
  local/calibrate_sre21av_v1.sh --cmd "$train_cmd" $score_plda_dir
  local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue ${score_plda_dir}_cal_v1
  local/score_sre_cts_superset.sh data/sre_cts_superset_16k_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio_dev_test audio_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio_eval_test audio_eval ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio-visual_eval_test audio-visual_eval ${score_plda_dir}_cal_v1
  local/score_sre21_official.sh $sre21_dev_root audio dev ${score_plda_dir}_cal_v1
  local/score_sre21_official.sh $sre21_eval_root audio eval ${score_plda_dir}_cal_v1
fi

score_plda_dir=$score_dir/plda_snorm_v1_${ncoh}

if [ $stage -le 5 ]; then
    #SRE superset and 16
    echo "SRE Superset Dev"
    steps_be/eval_be_plda_snorm_v2_cts.sh \
      --cmd "$train_cmd --mem 8G" \
      --plda_type $plda_type --ncoh $ncoh --num-parts 100 \
      data/sre_cts_superset_8k_dev/trials \
      data/sre_cts_superset_8k_dev/utt2enroll \
      $xvector_dir/sre_cts_superset_8k_dev/xvector.scp \
      data/sre_cts_superset_8k_dev/utt2lang \
      data/sre_cts_superset_8k_dev/utt2lang \
      $xvector_dir \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre_cts_superset_dev_scores &

    echo "SRE16"
    steps_be/eval_be_plda_snorm_v2_cts.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type --ncoh $ncoh \
      data/sre16_eval40_yue_test/trials \
      data/sre16_eval40_yue_enroll/utt2spk \
      $xvector_dir/sre16_eval40_yue/xvector.scp \
      data/sre16_eval40_yue_enroll/utt2lang \
      data/sre16_eval40_yue_test/utt2lang \
      $xvector_dir \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre16_eval40_yue_scores &
    
    wait

    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre_cts_superset.sh data/sre_cts_superset_8k_dev $score_plda_dir
fi


if [ $stage -le 6 ]; then

    #SRE21
    echo "SRE21 Audio Dev"
    steps_be/eval_be_plda_snorm_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
      data/sre21_audio_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio_dev/xvector.scp \
      data/sre21_audio_dev_enroll/utt2lang \
      data/sre21_audio_dev_test/utt2lang \
      data/sre21_audio_dev_enroll/segments.csv \
      data/sre21_audio_dev_test/segments.csv \
      $xvector_dir \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio_dev_scores &

    echo "SRE21 Audio-Visual Dev"
    steps_be/eval_be_plda_snorm_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
      data/sre21_audio-visual_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_dev/xvector.scp \
      data/sre21_audio_dev_enroll/utt2lang \
      data/sre21_audio-visual_dev_test/utt2lang \
      data/sre21_audio_dev_enroll/segments.csv \
      data/sre21_audio-visual_dev_test/segments.csv \
      $xvector_dir \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio-visual_dev_scores &

    echo "SRE21 Audio Eval"
    steps_be/eval_be_plda_snorm_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh --num-parts 40 \
      data/sre21_audio_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio_eval/xvector.scp \
      data/sre21_audio_eval_enroll/utt2est_lang \
      data/sre21_audio_eval_test/utt2est_lang \
      data/sre21_audio_eval_enroll/segments.csv \
      data/sre21_audio_eval_test/segments.csv \
      $xvector_dir \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio_eval_scores &

    echo "SRE21 Audio-Visual Eval"
    steps_be/eval_be_plda_snorm_v2_sre21.sh \
      --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh --num-parts 20 \
      data/sre21_audio-visual_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_eval/xvector.scp \
      data/sre21_audio_eval_enroll/utt2est_lang \
      data/sre21_audio-visual_eval_test/utt2est_lang \
      data/sre21_audio_eval_enroll/segments.csv \
      data/sre21_audio-visual_eval_test/segments.csv \
      $xvector_dir \
      $be_dir/cent_pca_lnorm \
      $be_dir/plda_adapt.h5 \
      $score_plda_dir/sre21_audio-visual_eval_scores &

    wait

    local/score_sre21.sh data/sre21_audio_dev_test audio_dev $score_plda_dir
    local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev $score_plda_dir
    local/score_sre21.sh data/sre21_audio_eval_test audio_eval $score_plda_dir
    local/score_sre21.sh data/sre21_audio-visual_eval_test audio-visual_eval $score_plda_dir
    local/score_sre21_official.sh $sre21_dev_root audio dev $score_plda_dir
    local/score_sre21_official.sh $sre21_eval_root audio eval $score_plda_dir
fi

if [ $stage -le 7 ];then
  local/calibrate_sre21av_v1.sh --cmd "$train_cmd" $score_plda_dir
  local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue ${score_plda_dir}_cal_v1
  local/score_sre_cts_superset.sh data/sre_cts_superset_16k_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio_dev_test audio_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio_eval_test audio_eval ${score_plda_dir}_cal_v1
  local/score_sre21.sh data/sre21_audio-visual_eval_test audio-visual_eval ${score_plda_dir}_cal_v1
  local/score_sre21_official.sh $sre21_dev_root audio dev ${score_plda_dir}_cal_v1
  local/score_sre21_official.sh $sre21_eval_root audio eval ${score_plda_dir}_cal_v1

fi

