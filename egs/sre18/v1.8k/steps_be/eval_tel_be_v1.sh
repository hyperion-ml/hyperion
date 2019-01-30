#!/bin/bash

cmd=run.pl
plda_type=frplda

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

ndx_file=$1
enroll_file=$2
vector_file=$3
preproc_file=$4
plda_file=$5
output_file=$6

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

hyp_enroll_file=$output_file.enroll
if [ ! -f $hyp_enroll_file ];then
    awk '{ print $2"="$1}' $enroll_file > $hyp_enroll_file
fi

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

$cmd $output_dir/log/${name}.log \
     python steps_be/eval-tel-be-v1.py \
     --iv-file scp:$vector_file \
     --ndx-file $hyp_ndx_file \
     --enroll-file $hyp_enroll_file \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --score-file $output_file


