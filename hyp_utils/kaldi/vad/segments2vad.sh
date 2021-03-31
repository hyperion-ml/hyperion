#!/bin/bash
# Copyright 2018   Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

cmd=run.pl
nj=20
stage=0

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [  $# != 3 ]; then
    echo "$0 <data-dir> <segments-dir> <binary-vad-dir>"
    exit 1
fi

data_dir=$1
segments_dir=$2
vad_dir=$3

mkdir -p $vad_dir/tmp

for f in utt2num_frames; do
  if [ ! -f $data_dir/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
done

name=$(basename $data_dir)

utils/split_data.sh $data_dir $nj || exit 1;
sdata_dir=$data_dir/split$nj;
segments=$segments_dir/segments

if [ $stage -le 1 ]; then
    mkdir -p $vad_dir/log

    $cmd JOB=1:$nj $vad_dir/log/vad_$name.JOB.log \
	 python steps_fe/segments2vad.py --segments $segments \
	 --num-frames $sdata_dir/JOB/utt2num_frames \
	 --vad-file $vad_dir/tmp/vad_${name}.JOB.ark
fi


if [ $stage -le 2 ];then
    for((j=1;j<=$nj;j++))
    do
	copy-vector ark:$vad_dir/tmp/vad_${name}.$j.ark \
		    ark,scp:$vad_dir/vad_${name}.$j.ark,$vad_dir/vad_${name}.$j.scp
	cat $vad_dir/vad_${name}.$j.scp || exit 1
    done > $vad_dir/vad_${name}.scp
    cp $vad_dir/vad_${name}.scp $data_dir/vad.scp
fi

