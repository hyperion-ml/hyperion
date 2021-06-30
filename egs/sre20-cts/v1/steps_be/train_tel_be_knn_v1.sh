#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
lda_dim=150
plda_type=frplda
y_dim=125
z_dim=150
knn=200
nj=40
set -e

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 <vector-file-trn> <data-dir-trn> <vector-file-enr> <data-dir-enr> <output-dir>"
  exit 1;
fi

vector_file_trn=$1
data_dir_trn=$2
vector_file_enr=$3
data_dir_enr=$4
output_dir=$5

mkdir -p $output_dir/log

for f in utt2spk; do
  if [ ! -f $data_dir_trn/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
  if [ ! -f $data_dir_enr/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
done

train_list=$output_dir/train_utt2spk

awk -v fv=$vector_file_trn 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $1,$2}}' $data_dir_trn/utt2spk > $train_list

enr_list=$data_dir_enr/utt2spk

$cmd JOB=1:$nj $output_dir/log/train_be.JOB.log \
     hyp_utils/conda_env.sh steps_be/train-tel-be-knn-v1.py \
     --v-file-train scp:$vector_file_trn \
     --train-list $train_list \
     --v-file-enroll-test scp:$vector_file_enr \
     --enroll-test-list $enr_list \
     --lda-dim $lda_dim \
     --plda-type $plda_type \
     --y-dim $y_dim --z-dim $z_dim --k-nn $knn \
     --output-path $output_dir \
     --part-idx JOB --num-parts $nj




     
