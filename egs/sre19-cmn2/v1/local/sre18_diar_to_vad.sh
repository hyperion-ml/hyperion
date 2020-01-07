#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir> <path-to-vad-dir>";
   echo "e.g.: $0 data/train exp/make_vad mfcc"
   exit 1;
fi

data=$1
logdir=$2
vaddir=$3

# make $vaddir an absolute pathname.
vaddir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $vaddir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $vaddir || exit 1;
mkdir -p $logdir || exit 1;


python local/sre18_diar_to_vad.py \
       $data/diarization $data/utt2num_frames | copy-vector ark:- ark,scp:$vaddir/vad_${name}.ark,$vaddir/vad_${name}.scp

cp $vaddir/vad_${name}.scp $data/vad.scp



