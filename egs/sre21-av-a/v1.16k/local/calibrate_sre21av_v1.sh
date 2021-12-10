#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

set -e

cmd=run.pl
p_tel=0.05
l2_reg=1e-4
sre_weight=0.1
extra_args=""
extra_tag=""

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
  echo "Usage: $0 <score-dir>"
  exit 1;
fi

score_dir=$1
cal_score_dir=${score_dir}_cal_v1
if [ -n "$extra_tag" ];then
  cal_score_dir=${cal_score_dir}_${extra_tag}
fi

mkdir -p $cal_score_dir

echo "$0 train calibration v1 for $score_dir"
awk '{ print $1,NF-1}' \
    data/sre16_eval40_yue_enroll/spk2utt \
    > $cal_score_dir/sre16_nenr

if [ -d data/sre_cts_superset_8k_dev ];then
  superset=sre_cts_superset_8k_dev
else
  superset=sre_cts_superset_16k_dev
fi

model_file=$cal_score_dir/cal_tel.h5
$cmd $cal_score_dir/train_cal_tel.log \
     hyp_utils/conda_env.sh \
     steps_be/train-calibration-v1.py \
     --score-file-sre16 $score_dir/sre16_eval40_yue_scores \
     --score-file-sre $score_dir/sre_cts_superset_dev_scores \
     --score-file-sre21 $score_dir/sre21_audio_dev_scores \
     --key-file-sre16 data/sre16_eval40_yue_test/trials \
     --key-file-sre data/$superset/trials \
     --key-file-sre21 data/sre21_audio_dev_test/trials \
     --nenr-sre16 $cal_score_dir/sre16_nenr \
     --model-file $model_file \
     --prior $p_tel --lambda-reg $l2_reg --sre-weight $sre_weight $extra_args

echo "$0 eval calibration v1 for sre16-yue"
$cmd $cal_score_dir/eval_cal_sre16_yue.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-calibration-v1-sre16.py \
     --in-score-file $score_dir/sre16_eval40_yue_scores \
     --ndx-file data/sre16_eval40_yue_test/trials \
     --nenr-file $cal_score_dir/sre16_nenr \
     --model-file $model_file \
     --out-score-file $cal_score_dir/sre16_eval40_yue_scores &

echo "$0 eval calibration v1 for sre-superset"
$cmd $cal_score_dir/eval_cal_sre_superset.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-calibration-v1-sre-superset.py \
     --in-score-file $score_dir/sre_cts_superset_dev_scores \
     --ndx-file data/$superset/trials \
     --model-file $model_file \
     --out-score-file $cal_score_dir/sre_cts_superset_dev_scores &

echo "$0 eval calibration v1 for sre21-audio-dev"
$cmd $cal_score_dir/eval_cal_sre21_audio_dev.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-calibration-v1-sre21-dev.py \
     --in-score-file $score_dir/sre21_audio_dev_scores \
     --ndx-file data/sre21_audio_dev_test/trials \
     --model-file $model_file \
     --out-score-file $cal_score_dir/sre21_audio_dev_scores &

echo "$0 eval calibration v1 for sre21-audio-visual-dev"
$cmd $cal_score_dir/eval_cal_sre21_audio-visual_dev.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-calibration-v1-sre21-dev.py \
     --in-score-file $score_dir/sre21_audio-visual_dev_scores \
     --ndx-file data/sre21_audio-visual_dev_test/trials \
     --model-file $model_file \
     --out-score-file $cal_score_dir/sre21_audio-visual_dev_scores &


utils/utt2spk_to_spk2utt.pl \
  data/sre21_audio_eval_enroll/utt2model | \
  awk '{ print $1,NF-1}' \
    > $cal_score_dir/sre21eval_nenr

awk -v u2l=data/sre21_audio_eval_enroll/utt2est_lang '
BEGIN{
  while(getline < u2l)
  {
     lang[$1]=$2
  }
}
{ if(!($2 in done)){ 
    print $2,lang[$1]; 
    done[$2]=1;
   } 
}' data/sre21_audio_eval_enroll/utt2model > $cal_score_dir/sre21eval_e2l

echo "$0 eval calibration v1 for sre21-audio-eval"
$cmd $cal_score_dir/eval_cal_sre21_audio_eval.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-calibration-v1-sre21-eval.py \
     --in-score-file $score_dir/sre21_audio_eval_scores \
     --ndx-file data/sre21_audio_eval_test/trials \
     --model-file $model_file \
     --out-score-file $cal_score_dir/sre21_audio_eval_scores \
     --nenr-file $cal_score_dir/sre21eval_nenr \
     --enroll-lang $cal_score_dir/sre21eval_e2l \
     --test-lang data/sre21_audio_eval_test/utt2est_lang &

echo "$0 eval calibration v1 for sre21-audio-eval"
$cmd $cal_score_dir/eval_cal_sre21_audio-visual_eval.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-calibration-v1-sre21-eval.py \
     --in-score-file $score_dir/sre21_audio-visual_eval_scores \
     --ndx-file data/sre21_audio-visual_eval_test/trials \
     --model-file $model_file \
     --out-score-file $cal_score_dir/sre21_audio-visual_eval_scores \
     --nenr-file $cal_score_dir/sre21eval_nenr \
     --enroll-lang $cal_score_dir/sre21eval_e2l \
     --test-lang data/sre21_audio-visual_eval_test/utt2est_lang &

wait

