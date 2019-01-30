#!/bin/bash

cmd=run.pl
plda_type=frplda

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

ndx_file=$1
diar_ndx_file=$2
enroll_file=$3
vector_file=$4
diar2orig=$5
preproc_file=$6
plda_file=$7
output_file=$8

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

hyp_enroll_file=${output_file}.enroll
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

NF=$(awk '{ c=NF } END{ print c}' $diar_ndx_file)
if [ $NF -eq 3 ];then
    # ndx file is is actuall key file, creates ndx
    hyp_diar_ndx_file=$output_file.diar_ndx
    if [ ! -f $hyp_diar_ndx_file ]; then
	awk '{ print $1,$2}' $diar_ndx_file > $hyp_diar_ndx_file
    fi
else
    hyp_diar_ndx_file=$diar_ndx_file
fi


$cmd $output_dir/log/${name}.log \
     python steps_be/eval-vid-be-diar-v1.py \
     --iv-file scp:$vector_file \
     --ndx-file $hyp_ndx_file \
     --diar-ndx-file $hyp_diar_ndx_file \
     --enroll-file $hyp_enroll_file \
     --diar2orig $diar2orig \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --score-file $output_file


