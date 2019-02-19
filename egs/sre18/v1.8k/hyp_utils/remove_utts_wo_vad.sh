#!/bin/bash
# Copyright 2018   Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

if [ -f path.sh ]; then . ./path.sh; fi
#. parse_options.sh || exit 1;

if [  $# != 1 ]; then
    echo "$0 <data-dir>"
    exit 1
fi

data_dir=$1

num_utts=$(wc -l $data_dir/utt2spk | awk '{ print $1}')
num_vad=$(wc -l $data_dir/vad.scp | awk '{ print $1}')

if [ $num_utts -gt $num_vad ];then
    echo "$0 utt $num_utts -> $num_vad"
    awk -v fvad=$data_dir/vad.scp 'BEGIN{ while(getline < fvad){ v[$1]=1}}{ if($1 in v){ print $0}}' $data_dir/wav.scp > $data_dir/wav.scp.tmp
    mv $data_dir/wav.scp.tmp $data_dir/wav.scp
    
    utils/fix_data_dir.sh $data_dir
fi
