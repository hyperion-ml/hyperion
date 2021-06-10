#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl
nj=20
stage=0
min_dur=10

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data-in-dir> <rttm-file> <data-out-dir> <vad-dir>"
  exit 1;
fi

data_dir=$1
rttm_file=$2
data_out_dir=$3
vad_dir=$4

mkdir -p $data_out_dir || exit 1;
mkdir -p $vad_dir/tmp

for f in wav.scp feats.scp utt2num_frames utt2spk; do
  if [ ! -f $data_dir/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
done

name=$(basename $data_out_dir)

utils/split_data.sh $data_dir $nj || exit 1;
sdata_dir=$data_dir/split$nj;

if [ $stage -le 1 ]; then
    rm -rf $data_out_dir/* 2>/dev/null
    mkdir -p $data_out_dir/log
    $cmd JOB=1:$nj $data_out_dir/log/rttm2vad.JOB.log \
	 python local/rttm2vad.py --rttm $rttm_file \
	 --num-frames $sdata_dir/JOB/utt2num_frames \
	 --vad-file $vad_dir/tmp/vad_${name}.JOB.ark \
	 --utt2orig $data_out_dir/utt2orig.JOB \
	 --ext-segments $data_out_dir/ext_segments.segments.JOB \
	 --min-dur $min_dur
    
    for((j=1;j<=$nj;j++))
    do
	cat $data_out_dir/utt2orig.$j || exit 1
    done > $data_out_dir/utt2orig
    rm $data_out_dir/utt2orig.*

    for((j=1;j<=$nj;j++))
    do
	for f in $(awk '{ print $2}' $data_out_dir/ext_segments.segments.$j | sort -u);
	do
	    awk '$2 == "'$f'"'  $data_out_dir/ext_segments.segments.$j | sort -g -k3;
	done
    done > $data_out_dir/ext_segments.segments
    rm $data_out_dir/ext_segments.segments.*

fi


if [ $stage -le 2 ];then
    for((j=1;j<=$nj;j++))
    do
	copy-vector ark:$vad_dir/tmp/vad_${name}.$j.ark \
		    ark,scp:$vad_dir/vad_${name}.$j.ark,$vad_dir/vad_${name}.$j.scp
	cat $vad_dir/vad_${name}.$j.scp || exit 1
    done > $vad_dir/vad_${name}.scp
    cp $vad_dir/vad_${name}.scp $data_out_dir/vad.scp
fi


if [ $stage -le 3 ];then
    for f in wav.scp feats.scp utt2num_frames utt2spk
    do
	file_in=$data_dir/$f
	file_out=$data_out_dir/$f
	awk -v f=$file_in 'BEGIN{
           while(getline < f)
           {
               k=$1;
               $1="";
               record[k]=$0;
           }
           OFS="";
        }
        { print $1, record[$2] }' $data_out_dir/utt2orig > $file_out
    done
    utils/utt2spk_to_spk2utt.pl $data_out_dir/utt2spk > $data_out_dir/spk2utt
    utils/utt2spk_to_spk2utt.pl $data_out_dir/utt2orig > $data_out_dir/orig2utt
fi

