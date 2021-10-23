#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
plda_type=frplda
ncoh=100
ncoh_discard=0
num_parts=1

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 11 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <enroll-utt2lang> <test-utt2lang> <enroll-segs-table> <test-segs-table> <cohort-vector-dir> <preproc-file-basename> <plda-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
enroll_u2l=$4
test_u2l=$5
enroll_tab=$6
test_tab=$7
coh_vector_dir=$8
preproc_file_bn=$9
plda_file=${10}
output_file=${11}

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

for((i=1;i<=$num_parts;i++));
do
  $cmd $output_dir/log/${name}_${i}.log \
       hyp_utils/conda_env.sh \
       steps_be/eval-be-plda-snorm-v2.py \
       --v-file scp:$vector_file \
       --ndx-file $ndx_file \
       --enroll-file $enroll_file \
       --preproc-file-basename $preproc_file_bn \
       --source-type mixed \
       --enroll-table $enroll_tab \
       --test-table $test_tab \
       --enroll-lang $enroll_u2l \
       --test-lang $test_u2l \
       --model-file $plda_file \
       --coh-v-dir $coh_vector_dir \
       --plda-type $plda_type \
       --coh-nbest $ncoh \
       --coh-nbest-discard $ncoh_discard \
       --score-file $output_file \
       --model-part-idx $i --num-model-parts $num_parts &
done
wait

if [ $num_parts -ne 1 ];then
  for((i=1;i<=$num_parts;i++));
  do
    for((j=1;j<=1;j++));
    do
      cat $output_file-$(printf "%03d" $i)-$(printf "%03d" $j)
    done
  done | sort -u > $output_file
fi

