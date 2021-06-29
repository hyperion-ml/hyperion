#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

set -e

cmd=run.pl
p_tel=0.05
l2_reg=1e-4

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
  echo "Usage: $0 <cal-set> <tel-score-dir>"
  exit 1;
fi

cal_set=$1
score_dir=$2

cal_score_dir=${score_dir}_cal_v1${cal_set}

mkdir -p $cal_score_dir

echo "$0 train calibration on ${cal_set}"

train_scores=$cal_score_dir/train_cal_scores
train_key=$cal_score_dir/train_cal_key

case "$cal_set" in
    sre16-yue) cp $score_dir/sre16_eval40_yue_scores $train_scores
	       cp data/sre16_eval40_yue_test/trials $train_key
	       ;;
    sre16-tgl) cp $score_dir/sre16_eval40_tgl_scores $train_scores
	       cp data/sre16_eval40_tgl_test/trials $train_key
	       ;;
    sre19) cp $score_dir/sre19_eval_cmn2_scores $train_scores
	   cp data/sre19_eval_test_cmn2/trials $train_key
	   ;;
    sre16) cat $score_dir/sre16_eval40_{yue,tgl}_scores > $train_scores
	   cat data/sre16_eval40_{yue,tgl}_test/trials > $train_key
	   ;;
    sre16-9) cat $score_dir/sre19_eval_cmn2_scores $score_dir/sre16_eval40_{yue,tgl}_scores > $train_scores
	     cat data/sre19_eval_test_cmn2/trials data/sre16_eval40_{yue,tgl}_test/trials > $train_key
	     ;;
    sre16-yue-9) cat $score_dir/sre19_eval_cmn2_scores $score_dir/sre16_eval40_yue_scores > $train_scores
	     cat data/sre19_eval_test_cmn2/trials data/sre16_eval40_yue_test/trials > $train_key
	     ;;
    *) echo "unknown calibration set $cal_set"
       exit 1
       ;;
esac


model_file=$cal_score_dir/cal_tel.h5
$cmd $cal_score_dir/train_cal_tel.log \
     hyp_utils/conda_env.sh steps_be/train-calibration-v1.py \
     --score-file $train_scores \
     --key-file $train_key \
     --model-file $model_file \
     --prior $p_tel --lambda-reg $l2_reg

ndxs=(sre16_eval40_yue_test sre16_eval40_tgl_test sre19_eval_test_cmn2 sre20cts_eval_test)
scores=(sre16_eval40_yue sre16_eval40_tgl sre19_eval_cmn2 sre20cts_eval)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do
    echo "$0 eval calibration on ${scores[$i]}"
    scores_in=$score_dir/${scores[$i]}_scores
    scores_out=$cal_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}/trials
    $cmd $cal_score_dir/eval_cal_${scores[$i]}.log \
	 hyp_utils/conda_env.sh steps_be/eval-calibration-v1.py \
	 --in-score-file $scores_in \
	 --ndx-file $ndx \
	 --model-file $model_file \
	 --out-score-file $scores_out &

done
wait





