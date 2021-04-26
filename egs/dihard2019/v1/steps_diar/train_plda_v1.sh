#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
lda_dim=150
plda_type=frplda
y_dim=100
z_dim=150
plda_opts=""

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 3 ]; then
  echo "Usage: $0 <xvector-scp-file> <data-dir> <output-dir>"
  exit 1;
fi

vector_file=$1
data_dir=$2
output_dir=$3

mkdir -p $output_dir/log

for f in utt2spk; do
  if [ ! -f $data_dir/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
done

train_list=$output_dir/train_utt2spk

#filter out the utterances that didn't got an x-vector (empty utts)
awk -v fv=$vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $1,$2}}' $data_dir/utt2spk > $train_list


$cmd $output_dir/log/train_plda.log \
    hyp_utils/conda_env.sh \
    steps_diar/train-plda-v1.py \
     --v-file scp:$vector_file \
     --train-list $train_list \
     --lda-dim $lda_dim \
     --plda-type $plda_type \
     --y-dim $y_dim --z-dim $z_dim \
     --output-path $output_dir $plda_opts




     
