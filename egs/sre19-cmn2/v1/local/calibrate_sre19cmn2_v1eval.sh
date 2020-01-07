#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

set -e

cmd=run.pl
p_tel=0.01

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
  echo "Usage: $0 <tel-score-dir>"
  exit 1;
fi

tel_score_dir=$1

cal_tel_score_dir=${tel_score_dir}_cal_v1eval

mkdir -p $cal_tel_score_dir

echo "$0 train calibration on sre18 CMN2 eval"

model_file=$cal_tel_score_dir/cal_tel.h5
train_scores=$tel_score_dir/sre18_eval_cmn2_scores
train_key=data/sre18_eval_test_cmn2/trials

$cmd $cal_tel_score_dir/train_cal_tel.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $p_tel 

ndxs=(sre18_dev_test_cmn2/trials sre18_eval_test_cmn2/trials sre19_eval_test_cmn2/trials)
scores=(sre18_dev_cmn2 sre18_eval_cmn2 sre19_eval_cmn2)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    scores_in=$tel_score_dir/${scores[$i]}_scores
    scores_out=$cal_tel_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    $cmd $cal_tel_score_dir/eval_cal_${scores[$i]}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

done
wait





