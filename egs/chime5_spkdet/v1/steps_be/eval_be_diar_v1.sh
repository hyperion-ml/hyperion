#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
plda_type=frplda

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e 

if [ $# -ne 7 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <enroll-vector-file> <test-vector-file> <preproc-file> <plda-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
enroll_vector_file=$3
test_vector_file=$4
preproc_file=$5
plda_file=$6
output_file=$7

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
     hyp_utils/conda_env.sh steps_be/eval-vid-be-diar-v2.py \
     --enroll-v-file scp:$enroll_vector_file \
     --test-v-file scp:$test_vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --score-file $output_file


