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
num_spks_unlab=1000
w_coral_mu=0.5
w_coral_T=0.75

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

vector_file=$1
data_dir=$2
adapt_vector_file=$3
adapt_data_dir=$4
unlab_vector_file=$5
unlab_meta=$6
output_dir=$7

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
unlab_list=$output_dir/unlab_utt2spk


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


awk -v fv=$unlab_vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
/unlabeled/ { if($1 in files){print $1,$5}}' $unlab_meta > $unlab_list

$cmd $output_dir/log/train_be.log \
    hyp_utils/conda_env.sh steps_be/train-tel-be-v3.py \
    --iv-file scp:$vector_file \
    --train-list $train_list \
    --adapt-iv-file scp:$adapt_vector_file \
    --adapt-list $adapt_list \
    --unlab-adapt-iv-file scp:$unlab_vector_file \
    --unlab-adapt-list $unlab_list \
    --lda-dim $lda_dim \
    --plda-type $plda_type \
    --y-dim $y_dim --z-dim $z_dim \
    --output-path $output_dir \
    --w-mu1 $w_mu1 --w-b1 $w_B1 --w-w1 $w_W1 \
    --w-mu2 $w_mu2 --w-b2 $w_B2 --w-w2 $w_W2 \
    --num-spks-unlab $num_spks_unlab --do-ahc \
    --w-coral-mu $w_coral_mu --w-coral-t $w_coral_T




     
