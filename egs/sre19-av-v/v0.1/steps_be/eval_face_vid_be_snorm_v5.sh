#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
#plda_type=frplda
ncoh=100
ncoh_discard=0
thr_ahc=0.8

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 8 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <ref-vector-file> <enr-vector-file> <test-vector-file> <cohort-list> <cohort-vector-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
ref_vector_file=$3
enr_vector_file=$4
test_vector_file=$5
coh_list=$6
coh_vector_file=$7
output_file=$8

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
     hyp_utils/conda_env.sh \
     steps_be/eval-face-vid-be-snorm-v5.py \
     --ref-v-file scp:$ref_vector_file \
     --enr-v-file scp:$enr_vector_file \
     --test-v-file scp:$test_vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --score-file $output_file \
     --coh-list $hyp_coh_list \
     --coh-v-file scp:$coh_vector_file \
     --coh-nbest $ncoh \
     --coh-nbest-discard $ncoh_discard \
     --thr-ahc $thr_ahc

rm -f $hyp_coh_list

