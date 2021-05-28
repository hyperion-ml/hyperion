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


echo "$0 calibrate on voices19"
scores_filename=voices19_challenge_dev_scores

for fold in 1 2
do
    if [ $fold -eq 1 ];then
	train_fold=2
    else
	train_fold=1
    fi

    
    echo "$0 train calibration on fold ${train_fold}"
    
    score_dir_f=${score_dir}_f${fold}
    cal_score_dir_f=${score_dir_f}_cal_v1

    model_file=$cal_score_dir_f/cal_voices19.h5
    train_scores=${score_dir}_f${train_fold}/$scores_filename
    train_key=data/voices19_challenge_dev_test_f${train_fold}/trials

    $cmd $cal_score_dir_f/train_cal_voices19.log \
	 steps_be/train-calibration-v1.py --score-file $train_scores \
	 --key-file $train_key --model-file $model_file --prior $prior


    echo "$0 eval calibration on fold ${fold}"
    
    scores_in=$score_dir_f/$scores_filename
    scores_out=$cal_score_dir_f/$scores_filename
    ndx=data/voices19_challenge_dev_test_f${fold}/trials
    $cmd $cal_score_dir_f/eval_cal_voices19_dev.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

done
wait

echo "$0 merge 2-fold calibrated scores for dev"
cal_score_dir=${score_dir}_2folds_cal_v1
mkdir -p $cal_score_dir
cat ${score_dir}_f{1,2}_cal_v1/$scores_filename > $cal_score_dir/$scores_filename

echo "$0 train calibration on folds 1+2"
score_dir_f=${score_dir}_2folds
cal_score_dir_f=${score_dir_f}_cal_v1

model_file=$cal_score_dir_f/cal_voices19.h5
train_scores=${score_dir_f}/$scores_filename
train_key=data/voices19_challenge_dev_test_2folds/trials

$cmd $cal_score_dir_f/train_cal_voices19.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $prior



echo "$0 eval calibration on eval"
scores_filename=voices19_challenge_eval_scores
scores_in=$score_dir/$scores_filename
scores_out=$cal_score_dir_f/$scores_filename
ndx=data/voices19_challenge_eval_test/trials
$cmd $cal_score_dir_f/eval_cal_voices19_eval.log \
     steps_be/eval-calibration-v1.py --in-score-file $scores_in \
     --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

