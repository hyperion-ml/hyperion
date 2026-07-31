#!/bin/bash

. path.sh

if [ $# -ne 3 ];then
  echo "Usage: $0 <score-dir> <model-file> <out"
  exit 1
fi

score_dirs="$1"
model_file=$2
output_dir=$3

dev_base=lre22_dev_scores.tsv
eval_base=lre22_eval_scores.tsv
dev_files=$(echo $score_dirs | awk 'BEGIN{OFS=","}{ for(i=1;i<=NF;i++){ $i="'\''"$i"/'$dev_base\''" }; print $0}')
eval_files=$(echo $score_dirs | awk 'BEGIN{OFS=","}{ for(i=1;i<=NF;i++){ $i="'\''"$i"/'$eval_base\''" }; print $0}')

dev_file_1=$(echo $dev_files | sed -e 's@'\''@@g' -e 's@,.*@@')
eval_file_1=$(echo $eval_files | sed -e 's@'\''@@g' -e 's@,.*@@')

dev_fus_file=$output_dir/$dev_base
eval_fus_file=$output_dir/$eval_base
mkdir -p $output_dir

if [ "$(hostname --domain)" == "cm.gemini" ];then
  module load matlab
fi

if [ -f $dev_file_1 ];then
  echo "
addpath('./steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
eval_fusion({$dev_files}, '$dev_fus_file', '$model_file');
" | matlab -nodisplay -nosplash > $output_dir/eval_lre22_dev.log
fi

if [ -f $eval_file_1 ];then
  echo "
addpath('./steps_be');
addpath(genpath('$PWD/focal_multiclass/v1.0'));
eval_fusion({$eval_files}, '$eval_fus_file', '$model_file');
" | matlab -nodisplay -nosplash > $output_dir/eval_lre22_eval.log
fi


