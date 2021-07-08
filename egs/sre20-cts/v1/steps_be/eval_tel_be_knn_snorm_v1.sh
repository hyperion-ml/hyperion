#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
plda_type=frplda
preproc_basename=lda_lnorm.h5
plda_basename=plda.h5
ncoh=100
num_parts=40

set -e

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 7 ]; then
  echo "Usage: $0 <ndx> <enroll-file> <vector-file> <cohort-list> <cohort-vector-file> <back-end-dir> <output-scores>"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
vector_file=$3
coh_list=$4
coh_vector_file=$5
backend_dir=$6
output_file=$7

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

for((i=1;i<=$num_parts;i++));
do
    $cmd $output_dir/log/${name}_${i}.log \
	 hyp_utils/conda_env.sh steps_be/eval-tel-be-knn-snorm-v1.py \
	 --iv-file scp:$vector_file \
	 --ndx-file $ndx_file \
	 --enroll-file $enroll_file \
	 --coh-list $hyp_coh_list \
	 --coh-iv-file scp:$coh_vector_file \
	 --back-end-dir $backend_dir \
	 --preproc-basename $preproc_basename \
	 --plda-basename $plda_basename \
	 --plda-type $plda_type \
	 --coh-nbest $ncoh \
	 --score-file $output_file \
	 --model-part-idx $i --num-model-parts $num_parts &
done
wait

if [ $num_parts -ne 1 ];then
    for((i=1;i<=$num_parts;i++));
    do
	for((j=1;j<=1;j++));
	do
	    cat $output_file-$(printf "%03d" $i)-$(printf "%03d" $j)
	done
    done | sort -u > $output_file
done
