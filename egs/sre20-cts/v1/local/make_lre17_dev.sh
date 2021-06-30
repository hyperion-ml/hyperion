#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [ $# != 3 ];then
    echo "Usage: $0 <lre17-dev-path> <fs 8/16> <output-path>"
    exit 1
fi

input_path=$1
fs=$2
output_path=$3

up=""
if [ $fs -eq 16 ];then
    up="sox -t wav - -t wav -e signed-integer -b 16 -r 16k - |"
fi

wav_path=$input_path/data/dev
key=$input_path/docs/lre17_dev_segments.key
for lang in ara-acm ara-apc ara-ary ara-arz \
		    por-brz qsl-pol qsl-rus \
		    spa-car spa-eur spa-lac \
		    zho-cmn zho-nan
do
    output_path_l=${output_path}_${lang}
    mkdir -p $output_path_l
    echo "Preparing LRE17 dev $lang"
    awk -v lang=$lang '$2==lang && $3=="mls14" && $4=="30" { 
key="lre17-"$1;
sub(/.*\//,"",key);
wav="'$wav_path'/"$1;
print key,"sph2pipe -f wav -p -c 1 "wav" |'"$up"'" }' $key | \
	sort -k1,1 > $output_path_l/wav.scp
    awk '{ print $1,$1}' $output_path_l/wav.scp > $output_path_l/utt2spk
    cp $output_path_l/utt2spk $output_path_l/spk2utt
    
    awk '{ print $1,"'$lang'"}' $output_path_l/utt2spk > $output_path_l/utt2lang

    utils/fix_data_dir.sh $output_path_l
    utils/validate_data_dir.sh --no-text --no-feats $output_path_l

done
