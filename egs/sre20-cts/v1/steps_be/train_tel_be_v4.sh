#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
lda_dim=150
plda_type=frplda
y_dim=100
z_dim=150
w_mu=1
w_B=0.5
w_W=0.5
w_coral_mu=0.5
w_coral_T=0.75
set -e

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 <vector-file> <data-dir> <adapt-vector-file> <adapt-data-dir> <output-dir>"
  exit 1;
fi

vector_file=$1
data_dir=$2
adapt_vector_file=$3
adapt_data_dir=$4
output_dir=$5

mkdir -p $output_dir/log

for f in utt2spk; do
  if [ ! -f $data_dir/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
done

for f in utt2spk; do
  if [ ! -f $adapt_data_dir/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
done

train_list=$output_dir/train_utt2spk
adapt_list=$output_dir/adapt_utt2spk

awk -v fv=$vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $1,$2}}' $data_dir/utt2spk > $train_list

awk -v fv=$adapt_vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $1,$2}}' $adapt_data_dir/utt2spk > $adapt_list

$cmd $output_dir/log/train_be.log \
    hyp_utils/conda_env.sh steps_be/train-tel-be-v4.py \
    --iv-file scp:$vector_file \
    --train-list $train_list \
    --adapt-iv-file scp:$adapt_vector_file \
    --adapt-list $adapt_list \
    --lda-dim $lda_dim \
    --plda-type $plda_type \
    --y-dim $y_dim --z-dim $z_dim \
    --output-path $output_dir \
    --w-mu $w_mu --w-b $w_B --w-w $w_W \
    --w-coral-mu $w_coral_mu --w-coral-t $w_coral_T
