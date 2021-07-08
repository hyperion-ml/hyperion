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

if [ $# -ne 9 ]; then
  echo "Usage: $0 <ndx> <enroll-file>  <enroll-vector-file> <test-vector-file> <cohort-list> <cohort-vector-file> <preproc-file> <plda-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
enroll_vector_file=$3
test_vector_file=$4
coh_list=$5
coh_vector_file=$6
preproc_file=$7
plda_file=$8
output_file=$9

output_dir=$(dirname $output_file)

mkdir -p $output_dir/log
name=$(basename $output_file)

hyp_coh_list=$output_file.cohort
awk -v fv=$coh_vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $1,$2}}' $coh_list > $hyp_coh_list


echo "$0 score $ndx_file"

$cmd $output_dir/log/${name}.log \
     hyp_utils/conda_env.sh steps_be/eval-vid-be-diar-snorm-v2.py \
     --enroll-v-file scp:$enroll_vector_file \
     --test-v-file scp:$test_vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --coh-list $hyp_coh_list \
     --coh-v-file scp:$coh_vector_file \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --coh-nbest $ncoh \
     --coh-nbest-discard $ncoh_discard \
     --score-file $output_file

rm -f $hyp_coh_list

