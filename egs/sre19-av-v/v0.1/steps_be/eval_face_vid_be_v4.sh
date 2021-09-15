#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
thr_ahc=0.9

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 6 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <ref-vector-file> <enr-vector-file> <test-vector-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
ref_vector_file=$3
enr_vector_file=$4
test_vector_file=$5
output_file=$6

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
    python steps_be/eval-face-vid-be-v4.py \
    --ref-v-file scp:$ref_vector_file \
    --enr-v-file scp:$enr_vector_file \
    --test-v-file scp:$test_vector_file \
    --ndx-file $ndx_file \
    --enroll-file $enroll_file \
    --score-file $output_file \
    --thr-ahc $thr_ahc

