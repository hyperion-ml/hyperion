#!/bin/bash
# Copyright 2018   Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

if [  $# != 4 ]; then
    echo "make_sitw_train.sh <SITW_PATH> <DEV/EVAL> <fs8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
subset=$2
fs=$3
output_path=$4


key=$input_path/$subset/keys/meta.lst
mkdir -p $output_path

awk '$8=="1" { sub(/.*\//,"",$1); sub(/\.flac$/,"",$1); 
print $1, $2, $3 }' $key > $output_path/key

wav_dir=$input_path/$subset/audio-wav-${fs}KHz/
awk '{ print "SITW-"$2"-"$1,"'$wav_dir'"$1".wav"}' \
    $output_path/key | sort > $output_path/wav.scp

awk '{ print "SITW-"$2"-"$1,$2 }' \
    $output_path/key | sort > $output_path/utt2spk

awk '{ print $2,$3 }' \
    $output_path/key | sort -u > $output_path/spk2gender

utils/utt2spk_to_spk2utt.pl $output_path/utt2spk > $output_path/spk2utt
utils/fix_data_dir.sh $output_path
utils/validate_data_dir.sh --no-text --no-feats $output_path









