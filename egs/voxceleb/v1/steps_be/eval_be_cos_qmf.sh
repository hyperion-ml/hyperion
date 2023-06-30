#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e
cmd=run.pl
stage=1
num_parts=16
coh_nbest=1000
preproc_file=""

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 8 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <numframes-file> <cohort-list> <cohort-vector-file> <qmf-weights> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
nf_file=$4
coh_file=$5
coh_v_file=$6
qmf_file=$7
output_file=$8

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

if [ -n "$preproc_file" ];then
  extra_args="--preproc-file $preproc_file"
fi

if [ $stage -le 1 ];then
  for((i=1;i<=$num_parts;i++));
  do
    for((j=1;j<=$num_parts;j++));
    do
      $cmd $output_dir/log/${name}_${i}_${j}.log \
	   hyp_utils/conda_env.sh \
	   steps_be/eval_be_cos_qmf.py $extra_args \
	   --v-file scp:$vector_file \
	   --ndx-file $ndx_file \
	   --enroll-file $enroll_file \
	   --score-file $output_file \
	   --num-frames-file $nf_file \
	   --coh-v-file scp:$coh_v_file \
	   --coh-file $coh_file \
	   --coh-nbest $coh_nbest \
	   --qmf-file $qmf_file \
	   --model-part-idx $i --num-model-parts $num_parts \
	   --seg-part-idx $j --num-seg-parts $num_parts &
    done
  done
  wait
fi


if [ $stage -le 2 ];then
  for suffix in "" _snorm _qmf
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


