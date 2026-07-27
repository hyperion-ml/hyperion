#!/bin/bash
# Copyright  2019   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
set -e
cmd=run.pl
nj=20
stage=1
frame_shift=10

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <rttm-file> <data-dir> <path-to-vad-dir>";
   echo "e.g.: $0 data/train/rttm data/train mfcc"
   exit 1;
fi

rttm=$1
data_dir=$2
vad_dir=$3

mkdir -p $vad_dir/log || exit 1

required="utt2num_frames"
for f in $required
do
    file=$data_dir/$f
    if [ ! -f "$file" ];then
	echo "$0: required file $file not found"
	exit 1
    fi
done

# make $vad_dir an absolute pathname.
vad_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $vad_dir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data_dir`
if [ $stage -le 1 ];then
    $cmd JOB=1:$nj $vad_dir/log/vad_$name.JOB.log \
	 rttm-to-bin-vad.py --rttm $rttm --num-frames $data_dir/utt2num_frames \
	 --frame-shift $frame_shift \
	 --output-path ark,scp:$vad_dir/vad_${name}.JOB.ark,$vad_dir/vad_${name}.JOB.scp \
	 --part-idx JOB --num-parts $nj
fi

if [ $stage -le 2 ];then
    for((j=1;j<=$nj;j++))
    do
	cat $vad_dir/vad_${name}.$j.scp || exit 1
    done > $vad_dir/vad_${name}.scp
    cp $vad_dir/vad_${name}.scp $data_dir/vad.scp
fi



