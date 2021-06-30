#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl
prior=0.01

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
  echo "Usage: $0 <score-dir>"
  exit 1;
fi

score_dir=$1

cal_score_dir=${score_dir}_cal_v1

mkdir -p $cal_score_dir

echo "$0 calibrate on voices19"

model_file=$cal_score_dir/cal_voices19.h5
train_scores=$score_dir/voices19_challenge_dev_scores
train_key=data/voices19_challenge_dev_test/trials

$cmd $cal_score_dir/train_cal_voices19.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $prior



echo "$0 eval calibration for voices19"
    
for s in dev eval
do
    scores_i=voices19_challenge_${s}_scores
    scores_in=$score_dir/$scores_i
    scores_out=$cal_score_dir/$scores_i
    ndx=data/voices19_challenge_${s}_test/trials
    $cmd $cal_score_dir/eval_cal_voices19_${s}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

done
wait

