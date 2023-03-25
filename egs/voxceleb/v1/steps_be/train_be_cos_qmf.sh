#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e
cmd=run.pl
stage=1
num_parts=8
coh_nbest=400

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 7 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <numframes-file> <cohort-list> <cohort-vector-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
nf_file=$4
coh_file=$5
coh_v_file=$6
output_file=$7

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

if [ $stage -le 1 ];then
  for((i=1;i<=$num_parts;i++));
  do
    for((j=1;j<=$num_parts;j++));
    do
      $cmd $output_dir/log/${name}_${i}_${j}.log \
	   hyp_utils/conda_env.sh \
	   steps_be/eval-be-cos-qmf.py \
	   --v-file scp:$vector_file \
	   --ndx-file $ndx_file \
	   --enroll-file $enroll_file \
	   --score-file $output_file \
	   --num-frames-file $nf_file \
	   --coh-v-file scp:$coh_v_file \
	   --coh-file $coh_file \
	   --coh-nbest $coh_nbest \
	   --model-part-idx $i --num-model-parts $num_parts \
	   --seg-part-idx $j --num-seg-parts $num_parts &
    done
  done
  wait
fi

if [ $stage -le 2 ];then
  for suffix in "" _maxnf _minnf _maxcohmu _mincohmu _snorm
  do
    output_file_k=${output_file}${suffix}
    for((i=1;i<=$num_parts;i++));
    do
      for((j=1;j<=$num_parts;j++));
      do
	cat $output_file_k-$(printf "%03d" $i)-$(printf "%03d" $j)
      done
    done | sort -u > $output_file_k
  done
fi

if [ $stage -le 3 ];then
  $cmd $output_dir/log/train_qmf_${name}.log \
       hyp_utils/conda_env.sh \
       steps_be/train-qmf.py \
       --score-file $output_file \
       --key-file $ndx_file \
       --model-file $output_dir/qmf.h5
fi


