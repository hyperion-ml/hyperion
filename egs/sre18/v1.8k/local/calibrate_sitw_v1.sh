#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl
prior=0.05

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
  echo "Usage: $0 <score-dir>"
  exit 1;
fi

score_dir=$1

cal_score_dir=${score_dir}_cal_v1

mkdir -p $cal_score_dir

echo "$0 calibrate on sitw core-multi"

model_file=$cal_score_dir/cal_sitw.h5
train_scores=$score_dir/sitw_dev_core-multi_scores
train_key=data/sitw_dev_test/trials/core-multi.lst

$cmd $cal_score_dir/train_cal_sitw.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $prior


conds=(core-core core-multi assist-core assist-multi)

n_conds=${#conds[*]}
for((i=0;i<$n_conds;i++))
do
    cond_i=${conds[$i]}
    echo "$0 eval calibration for sitw ${cond_i}"
    
    # SITW dev
    scores_i=sitw_dev_${cond_i}_scores
    scores_in=$score_dir/$scores_i
    scores_out=$cal_score_dir/$scores_i
    ndx=data/sitw_dev_test/trials/${cond_i}.lst
    $cmd $cal_score_dir/eval_cal_sitw_dev_${cond_i}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &


    # SITW eval
    scores_i=sitw_eval_${cond_i}_scores
    scores_in=$score_dir/$scores_i
    scores_out=$cal_score_dir/$scores_i
    ndx=data/sitw_eval_test/trials/${cond_i}.lst
    $cmd $cal_score_dir/eval_cal_sitw_eval_${cond_i}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

done

wait
