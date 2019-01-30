#!/bin/bash

cmd=run.pl
lda_dim=150
plda_type=frplda
y_dim=100
z_dim=150
w_mu1=1
w_B1=1
w_W1=1
w_mu2=1
w_B2=1
w_W2=1
num_spks=1000


if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

vector_file=$1
data_dir=$2
adapt_vector_file=$3
meta=$4
output_dir=$5

mkdir -p $output_dir/log

for f in utt2spk; do
  if [ ! -f $data_dir/$f ]; then
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
/unlabeled/ { if($1 in files){print $1,$5}}' $meta > $adapt_list

$cmd $output_dir/log/train_be.log \
     python steps_be/train-tel-be-v1.py \
     --iv-file scp:$vector_file \
     --train-list $train_list \
     --adapt-iv-file scp:$adapt_vector_file \
     --adapt-list $adapt_list \
     --lda-dim $lda_dim \
     --plda-type $plda_type \
     --y-dim $y_dim --z-dim $z_dim \
     --output-path $output_dir \
     --w-mu1 $w_mu1 --w-b1 $w_B1 --w-w1 $w_W1 \
     --w-mu2 $w_mu2 --w-b2 $w_B2 --w-w2 $w_W2 \
     --num-spks $num_spks --do-ahc




     
