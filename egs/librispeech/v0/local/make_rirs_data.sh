#!/bin/bash
#
# Copyright 2020 Johns Hopkins University (Jesus Villalba)
#           
# Apache 2.0.
set -e

if [ $# != 3 ]; then
  echo "Usage: $0 <rir-dir> <fs> <data-dir>"
  echo "e.g.: $0 RIRS_NOISES/simulated_rirs/smallroom 16 data/rirs_smallroom"
fi

rir_dir=$1
fs=$2
data_dir=$3

mkdir -p $data_dir

rir_list=$rir_dir/rir_list
if [ "$fs" -eq 16 ];then
    awk '{ key=$5; sub(/.*\//,"",key); print key,$5 }' $rir_list > $data_dir/wav.scp
else
    awk '{ 
key=$5; sub(/.*\//,"",key); 
print key,"sox "$5" -r 8000 -t wav -b 16 -e signed-integer - |" }' \
    $rir_list > $data_dir/wav.scp
fi
awk '{ key=$5; sub(/.*\//,"",key); print key,$4 }' $rir_list > $data_dir/rir2room

