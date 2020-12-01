#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 4 ]; then
    echo "Usage: $0 <babel-path> <lang-id> <fs 8/16> <OUTPATH>"
    exit 1
fi

input_path=$1
lang=$2
fs=$3
output_path=$4

echo "Preparing Babel $lang"
up=""
if [ $fs -eq 16 ];then
    up="sox -t wav - -t wav -e signed-integer -b 16 -r 16k - |"
fi

mkdir -p $output_path
input_path=$input_path/conversational

for d in training untranscribed-training dev eval
do
    if [ -d $input_path/$d/audio ];then
	find $input_path/$d/audio -name "*.sph" | sort | \
	    awk '{ 
key=$1;
sub(/.*\//,"",key);
print key,"sph2pipe -f wav -p -c 1 "$1" |'"$up"'" }'
    fi
done | sort -k1,1 > $output_path/wav.scp

awk '{ print $1,$1}' $output_path/wav.scp > $output_path/utt2spk
cp $output_path/utt2spk $output_path/spk2utt

awk '{ print $1,"'$lang'"}' $output_path/utt2spk > $output_path/utt2lang

utils/fix_data_dir.sh $output_path
utils/validate_data_dir.sh --no-text --no-feats $output_path


    


