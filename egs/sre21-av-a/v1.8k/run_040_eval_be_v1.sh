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

ncoh=500
coh_data=voxceleb_sre_alllangs
ft=1
pca_var_r=0.45
lda_dim=200
plda_y_dim=150
plda_z_dim=200

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

plda_data=voxceleb_sre_alllangs
#plda_data=sre_chnspks
plda_label=${plda_type}y${plda_y_dim}_v1
be_name=pca${pca_var_r}_lda${lda_dim}_${plda_label}_${plda_data}


xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda

if [ "$plda_data" == "sre_chnspks" ];then
  if [ ! -d data/sre_chnspks ];then
    local/filter_chnspks.sh data/sre_alllangs data/sre_chnspks
  fi
  if [ ! -L $xvector_dir/sre_chnspks ];then
    cd $xvector_dir
    ln -s sre_alllangs sre_chnspks
    cd -
  fi
fi

if [ $stage -le 1 ]; then
  echo "Train PLDA on $plda_data"
  steps_be/train_be_plda_v1.sh \
    --cmd "$train_cmd" \
    --lda_dim $lda_dim \
    --plda_type $plda_type \
    --y_dim $plda_y_dim --z_dim $plda_z_dim --pca-var-r $pca_var_r \
    $xvector_dir/$plda_data/xvector.scp \
    data/$plda_data \
    $be_dir

fi


if [ $stage -le 2 ]; then
    #SRE superset and 16
    echo "SRE Superset Dev"
    steps_be/eval_be_plda_v1.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      data/sre_cts_superset_8k_dev/trials \
      data/sre_cts_superset_8k_dev/utt2enroll \
      $xvector_dir/sre_cts_superset_8k_dev/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre_cts_superset_dev_scores &

    echo "SRE16"
    steps_be/eval_be_plda_v1.sh \
      --cmd "$train_cmd" \
      --plda_type $plda_type \
      data/sre16_eval40_yue_test/trials \
      data/sre16_eval40_yue_enroll/utt2spk \
      $xvector_dir/sre16_eval40_yue/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre16_eval40_yue_scores &
    
    wait

    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre_cts_superset.sh data/sre_cts_superset_8k_dev $score_plda_dir
fi


if [ $stage -le 3 ]; then

    #SRE21
    echo "SRE21 Audio Dev"
    steps_be/eval_be_plda_v1.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio_dev/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre21_audio_dev_scores &

    echo "SRE21 Audio-Visual Dev"
    steps_be/eval_be_plda_v1.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio-visual_dev_test/trials \
      data/sre21_audio_dev_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_dev/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre21_audio-visual_dev_scores &

    echo "SRE21 Audio Eval"
    steps_be/eval_be_plda_v1.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio_eval/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
      $score_plda_dir/sre21_audio_eval_scores &

    echo "SRE21 Audio-Visual Eval"
    steps_be/eval_be_plda_v1.sh \
      --cmd "$train_cmd" --plda_type $plda_type \
      data/sre21_audio-visual_eval_test/trials \
      data/sre21_audio_eval_enroll/utt2model \
      $xvector_dir/sre21_audio-visual_eval/xvector.scp \
      $be_dir/lda_lnorm.h5 \
      $be_dir/plda.h5 \
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
exit
