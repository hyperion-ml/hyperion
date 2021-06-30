#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
ncoh=100

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 6 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <cohort-list> <cohort-vector-file> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
coh_list=$4
coh_vector_file=$5
output_file=$6

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
     hyp_utils/conda_env.sh steps_be/eval-tel-be-snorm-v2.py \
     --iv-file scp:$vector_file \
     --ndx-file $ndx_file \
     --enroll-file $enroll_file \
     --coh-list $hyp_coh_list \
     --coh-iv-file scp:$coh_vector_file \
     --coh-nbest $ncoh \
     --score-file $output_file

rm $hyp_coh_list
