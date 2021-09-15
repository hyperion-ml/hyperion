#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

set -e

cmd=run.pl
p_vid=0.05
l2_reg=1e-5

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
  echo "Usage: $0 <score-dir>"
  exit 1;
fi

score_dir=$1

cal_av_score_dir=${score_dir}_cal_v1_sre19

mkdir -p $cal_av_score_dir

echo "$0 train calibration on sre19 AV dev"

model_file=$cal_av_score_dir/cal_av.h5
train_scores=$score_dir/sre19_av_v_dev_scores
train_key=data/sre19_av_v_dev_test/trials

$cmd $cal_av_score_dir/train_cal_av.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $p_vid --lambda-reg $l2_reg

ndxs=(sre19_av_v_dev_test/trials \
    sre19_av_v_eval_test/trials \
    janus_dev_test_core/trials \
    janus_eval_test_core/trials)
scores=(sre19_av_v_dev sre19_av_v_eval \
    janus_dev_core janus_eval_core)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do
    echo "$0 eval calibration on ${scores[$i]}"
    scores_in=$score_dir/${scores[$i]}_scores
    scores_out=$cal_av_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    $cmd $cal_av_score_dir/eval_cal_${scores[$i]}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

done
wait





