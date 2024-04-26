#!/bin/bash

. path.sh

if [ $# -ne 1 ];then
  echo "Usage: $0 <score-dir>"
  exit 1
fi

score_dir=$1
nocal_dir=$score_dir/nocal
cal_dir=$score_dir/cal_v1

train_list=data/lre22_dev/utt2lang
train_file=$nocal_dir/lre22_dev_scores.tsv
train_cal_file=$cal_dir/lre22_dev_scores.tsv
eval_file=$nocal_dir/lre22_eval_scores.tsv
eval_cal_file=$cal_dir/lre22_eval_scores.tsv
mkdir -p $cal_dir
model_file=$cal_dir/cal.mat

if [ "$(hostname --domain)" == "cm.gemini" ];then
  module load matlab
fi

echo "
addpath('steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
train_fusion('$train_list', {'$train_file'}, '$model_file');
" | matlab -nodisplay -nosplash > $cal_dir/train.log

echo "
addpath('./steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
eval_fusion({'$train_file'}, '$train_cal_file', '$model_file');
" | matlab -nodisplay -nosplash > $cal_dir/eval_lre22_dev.log

if [ -f $eval_file ];then
  echo "
addpath('./steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
eval_fusion({'$eval_file'}, '$eval_cal_file', '$model_file');
" | matlab -nodisplay -nosplash > $cal_dir/eval_lre22_eval.log
fi


