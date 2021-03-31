#!/bin/bash
# Copyright 2020   Johns Hopkins Universiy (Jesus Villalba)
# Apache 2.0.
#
# Creates the DIHARD 2019 data directories.

if [ $# != 2 ]; then
  echo "Usage: $0 <dihard-dir> <data-dir>"
  echo " e.g.: $0 /export/corpora/LDC/LDC2019E31 data/dihard2019_dev"
fi

dihard_dir=$1/data/single_channel
data_dir=$2

echo "making data dir $data_dir"

mkdir -p $data_dir

find $dihard_dir -name "*.flac" | \
    awk '
{ bn=$1; sub(/.*\//,"",bn); sub(/\.flac$/,"",bn); 
  print bn, "sox "$1" -t wav -b 16 -e signed-integer - |" }' | sort -k1,1 > $data_dir/wav.scp

awk '{ print $1,$1}' $data_dir/wav.scp  > $data_dir/utt2spk
cat $data_dir/utt2spk > $data_dir/spk2utt

for f in $(find $dihard_dir -name "*.lab" | sort)
do
    awk '{ bn=FILENAME; sub(/.*\//,"",bn); sub(/\.lab$/,"",bn); 
           printf "%s-%010d-%010d %s %f %f\n", bn, $1*1000, $2*1000, bn, $1, $2}' $f
done > $data_dir/vad.segments


rm -f $data_dir/reco2num_spks
for f in $(find $dihard_dir -name "*.rttm" | sort)
do
    cat $f
    awk '{ print $2, $8}' $f | sort -u | awk '{ f=$1; count++}END{ print f, count}' >> $data_dir/reco2num_spks

done > $data_dir/diarization.rttm

for f in $(find $dihard_dir -name "*.uem" | sort)
do
    cat $f
done > $data_dir/diarization.uem

utils/validate_data_dir.sh --no-feats --no-text $data_dir
    
