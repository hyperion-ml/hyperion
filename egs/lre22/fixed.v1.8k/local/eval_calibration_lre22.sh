#!/bin/bash

. path.sh

if [ $# -ne 2 ];then
  echo "Usage: $0 <score-dir> <model-file>"
  exit 1
fi

score_dir=$1
model_file=$2
nocal_dir=$score_dir/nocal
cal_dir=$score_dir/cal_v1

dev_file=$nocal_dir/lre22_dev_scores.tsv
dev_cal_file=$cal_dir/lre22_dev_scores.tsv
eval_file=$nocal_dir/lre22_eval_scores.tsv
eval_cal_file=$cal_dir/lre22_eval_scores.tsv
mkdir -p $cal_dir


if [ "$(hostname --domain)" == "cm.gemini" ];then
  module load matlab
fi

if [ -f $dev_file ];then
  echo "
addpath('./steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
eval_fusion({'$dev_file'}, '$dev_cal_file', '$model_file');
" | matlab -nodisplay -nosplash > $cal_dir/eval_lre22_dev.log
fi

if [ -f $eval_file ];then
  echo "
addpath('./steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
eval_fusion({'$eval_file'}, '$eval_cal_file', '$model_file');
" | matlab -nodisplay -nosplash > $cal_dir/eval_lre22_eval.log
fi


