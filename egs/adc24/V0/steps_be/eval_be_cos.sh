#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e
cmd=queue.pl
num_parts=8
preproc_file=""

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
echo $#
echo $1
echo $2
echo $3 
echo $4

if [ $# -ne 3 ]; then
  echo "Usage: $0 <ndx>  khra<enroll-file> <vector-file> <output-scores>"
  exit 1;
fi


enroll_file=$1
vector_file=$2
output_file=$3

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)



if [ -n "$preproc_file" ];then
  extra_args="--preproc-file $preproc_file"
fi

$cmd $output_dir/log/${name}_${i}_${j}.log \
  hyp_utils/conda_env.sh \
  steps_be/eval_be_cos.py $extra_args \
  --xvector_train_scp $enroll_file \
  --xvector_test_scp $vector_file \
  --output_file $score_cosine_dir/adi17_scores.json

 
wait
