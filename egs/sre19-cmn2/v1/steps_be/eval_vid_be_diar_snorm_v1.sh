#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
plda_type=frplda
ncoh=100
ncoh_discard=0

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -e

if [ $# -ne 10 ]; then
  echo "Usage: $0 <ndx> <diar-ndx> <enroll-file> <vector-file> <diar-segments-to-orig-utt> <cohort-list> <cohort-vector-file> <preproc-file> <plda-file> <output-scores>"
  exit 1;
fi


ndx_file=$1
diar_ndx_file=$2
enroll_file=$3
vector_file=$4
diar2orig=$5
coh_list=$6
coh_vector_file=$7
preproc_file=$8
plda_file=$9
output_file=${10}

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)


hyp_enroll_file=$enroll_file

hyp_coh_list=$output_file.cohort
awk -v fv=$coh_vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $1,$2}}' $coh_list > $hyp_coh_list


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

echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
     python steps_be/eval-vid-be-diar-snorm-v1.py \
     --iv-file scp:$vector_file \
     --ndx-file $hyp_ndx_file \
     --diar-ndx-file $hyp_diar_ndx_file \
     --enroll-file $hyp_enroll_file \
     --diar2orig $diar2orig \
     --coh-list $hyp_coh_list \
     --coh-iv-file scp:$coh_vector_file \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --coh-nbest $ncoh \
     --coh-nbest-discard $ncoh_discard \
     --score-file $output_file


