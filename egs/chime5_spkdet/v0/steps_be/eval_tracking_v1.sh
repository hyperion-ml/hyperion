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
  echo "Usage: $0 <ndx> <enroll-file> <ext-segments-file> <vector-file> <preproc-file> <plda-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
segment_file=$3
vector_file=$4
preproc_file=$5
plda_file=$6
output_file=$7

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

NF=$(awk '{ c=NF } END{ print c}' $ndx_file)
if [ $NF -eq 3 ];then
    # ndx file is is actuall key file, creates ndx
    hyp_ndx_file=$output_file.ndx
    if [ ! -f $hyp_ndx_file ]; then
	awk '{ print $1,$2}' $ndx_file > $hyp_ndx_file
    fi
else
    hyp_ndx_file=$ndx_file
fi

echo "$0 tracking $ndx_file"

$cmd $output_dir/log/${name}.log \
     python steps_be/eval-tracking-v1.py \
     --iv-file scp:$vector_file \
     --ndx-file $hyp_ndx_file \
     --enroll-file $enroll_file \
     --segments-file $segment_file \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --rttm-file $output_file


rm -f $output_file.ndx
