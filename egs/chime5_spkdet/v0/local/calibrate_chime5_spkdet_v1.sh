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

echo "$0 calibrate on chime5 close-talk"

model_file=$cal_score_dir/cal_chime5.h5
train_scores=$score_dir/chime5_spkdet_scores
train_key=data/chime5_spkdet_test/trials_BIN.SUM

$cmd $cal_score_dir/train_cal_chime5.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $prior



echo "$0 eval calibration for all chime5 conditions"
    
scores_i=chime5_spkdet_scores
scores_in=$score_dir/$scores_i
scores_out=$cal_score_dir/$scores_i
ndx=data/chime5_spkdet_test/trials
$cmd $cal_score_dir/eval_cal_chime5.log \
     steps_be/eval-calibration-v1.py --in-score-file $scores_in \
     --ndx-file $ndx --model-file $model_file --out-score-file $scores_out



