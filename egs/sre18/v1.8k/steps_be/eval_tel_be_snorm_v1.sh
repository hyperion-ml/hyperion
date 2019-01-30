#!/bin/bash

cmd=run.pl
plda_type=frplda
ncoh=100

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

ndx_file=$1
enroll_file=$2
vector_file=$3
coh_list=$4
coh_vector_file=$5
preproc_file=$6
plda_file=$7
output_file=$8

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

hyp_enroll_file=$output_file.enroll
if [ ! -f $hyp_enroll_file ];then
    awk '{ print $2"="$1}' $enroll_file > $hyp_enroll_file
fi

hyp_coh_list=$output_file.cohort
awk -v fv=$coh_vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $2"="$1}}' $coh_list > $hyp_coh_list

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
     python steps_be/eval-tel-be-snorm-v1.py \
     --iv-file scp:$vector_file \
     --ndx-file $hyp_ndx_file \
     --enroll-file $hyp_enroll_file \
     --coh-list $hyp_coh_list \
     --coh-iv-file scp:$coh_vector_file \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --coh-nbest $ncoh \
     --score-file $output_file


