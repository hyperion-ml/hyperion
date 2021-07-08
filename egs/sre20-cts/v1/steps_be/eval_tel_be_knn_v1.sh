#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
plda_type=frplda
preproc_basename=lda_lnorm.h5
plda_basename=plda.h5
set -e

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <back-end-dir> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
backend_dir=$4
output_file=$5

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
     hyp_utils/conda_env.sh steps_be/eval-tel-be-knn-v1.py \
     --iv-file scp:$vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --back-end-dir $backend_dir \
     --preproc-basename $preproc_basename \
     --plda-basename $plda_basename \
     --plda-type $plda_type \
     --score-file $output_file
