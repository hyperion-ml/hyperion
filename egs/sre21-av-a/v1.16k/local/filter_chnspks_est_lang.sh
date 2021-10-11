#!/bin/bash
#
# Copyright 2021 Johns Hopkins University (Jesus Villalba)
#           
# Apache 2.0.
#
# Creates a copy of the dataset with only the speakers that speaker Mandarin or Cantonese
# looking utt2est_lang instead of utt2lang
set -e

if [ $# != 2 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir>"
  echo "e.g.: $0 data/voxcelebcat data/voxcelebcat_chnspks"
fi

in_data_dir=$1
out_data_dir=$2

rm -rf $out_data_dir
cp -r $in_data_dir $out_data_dir

awk '$2 ~ "CMN" || $2 ~ "YUE" { print $1}' \
    $in_data_dir/utt2est_lang > $out_data_dir/chn_utts

awk -v utts=$out_data_dir/chn_utts 'BEGIN{
while(getline < utts){
  chn_utts[$0]=1;
}
}
{ if ($1 in chn_utts) { print $2}}' \
    $in_data_dir/utt2spk | sort -u > $out_data_dir/chn_spks

awk -v spks=$out_data_dir/chn_spks 'BEGIN{
while(getline < spks){
  chn_spks[$0]=1;
}
}
{ if ($2 in chn_spks) { print $0}}' \
    $in_data_dir/utt2spk | sort -u > $out_data_dir/utt2spk

utils/fix_data_dir.sh $out_data_dir

