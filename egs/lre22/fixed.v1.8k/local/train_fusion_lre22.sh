#!/bin/bash

. path.sh

if [ $# -ne 2 ];then
  echo "Usage: $0 <score-dirs> <output-dir>"
  exit 1
fi

score_dirs="$1"
output_dir=$2

train_list=data/lre22_dev/utt2lang
train_base=lre22_dev_scores.tsv
train_files=$(echo $score_dirs | awk 'BEGIN{OFS=","}{ for(i=1;i<=NF;i++){ $i="'\''"$i"/'$train_base\''" }; print $0}')

train_fus_file=$output_dir/$train_base
mkdir -p $output_dir
model_file=$output_dir/fus.mat

if [ "$(hostname --domain)" == "cm.gemini" ];then
  module load matlab
fi

echo "
addpath('steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
train_fusion('$train_list', {$train_files}, '$model_file');
" | matlab -nodisplay -nosplash > $output_dir/train.log

echo "
addpath('./steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
eval_fusion({$train_files}, '$train_fus_file', '$model_file');
" | matlab -nodisplay -nosplash > $output_dir/eval_lre22_dev.log

