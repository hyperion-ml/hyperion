#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

set -e

cmd=run.pl
prior=0.05
l2_reg=1e-5

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
  echo "Usage: $0 <cal-set> <score-dir>"
  exit 1;
fi

score_dir=$1
cal_score_dir=${score_dir}_cal_v1

mkdir -p $cal_score_dir

echo "$0 train calibration on VoxCeleb1 Original Clean"

model_file=$cal_score_dir/cal_tel.h5
train_scores=$score_dir/voxceleb1_scores
train_key=data/voxceleb1_test/trials_o_clean

$cmd $cal_score_dir/train_cal_tel.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $prior --lambda-reg $l2_reg

ndxs=(voxceleb1_test/trials_o_clean)
scores=(voxceleb1)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do
    echo "$0 eval calibration on ${scores[$i]}"
    scores_in=$score_dir/${scores[$i]}_scores
    scores_out=$cal_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    $cmd $cal_score_dir/eval_cal_${scores[$i]}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

done
wait





