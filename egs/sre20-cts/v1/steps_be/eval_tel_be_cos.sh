#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
set -e

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
output_file=$4

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
     hyp_utils/conda_env.sh steps_be/eval-tel-be-v2.py \
     --iv-file scp:$vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --score-file $output_file


