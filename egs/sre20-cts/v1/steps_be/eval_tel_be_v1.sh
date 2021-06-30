#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
plda_type=frplda
set -e

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <preproc-file> <plda-file> <output-scores>"
  exit 1;
fi


ndx_file=$1
enroll_file=$2
vector_file=$3
preproc_file=$4
plda_file=$5
output_file=$6

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
     hyp_utils/conda_env.sh steps_be/eval-tel-be-v1.py \
     --iv-file scp:$vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --score-file $output_file


