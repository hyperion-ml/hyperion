#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
plda_type=frplda

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 8 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <enroll-utt2lang> <test-utt2lang> <preproc-file-basename> <plda-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
enroll_u2l=$4
test_u2l=$5
preproc_file_bn=$6
plda_file_bn=$7
output_file=$8

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-be-plda-v3.py \
     --v-file scp:$vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --preproc-file-basename $preproc_file_bn \
     --source-type cts \
     --enroll-lang $enroll_u2l \
     --test-lang $test_u2l \
     --model-file-basename $plda_file_bn \
     --plda-type $plda_type \
     --score-file $output_file
