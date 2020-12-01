#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 6 ]; then
    echo "Usage: $0 <CV-PATH> <lang-id> <min-utt-per-spk> <num-utt-to-cat> <fs 8/16> <OUTPATH>"
    exit 1
fi

input_path=$1
lang=$2
min_utts=$3
num_cat=$4
fs=$5
output_path=$6

echo "Preparing CommonVoice $lang"
down=""
if [ $fs -eq 8 ];then
    down=" -r 8k "
else
    down=" -r 16k "
fi

input_path=$input_path/$lang
mkdir -p $output_path/lists_cat

list=$input_path/validated.tsv

echo "Filtering Spks with less than $min_utts"

awk '!/client_id/ { print $2,$1 }' $list > $output_path/utt2client
utils/utt2spk_to_spk2utt.pl $output_path/utt2client > $output_path/client2utt

awk -v min_utts=$min_utts '
NF>min_utts { count++; printf "%s cv-'$lang'-id%06d %d\n", $1, count, NF-1; }' \
    $output_path/client2utt > $output_path/client2spk2num_utts

awk -v c2s=$output_path/client2spk2num_utts  'BEGIN{
while(getline < c2s)
{
    spks[$1]=$2
}
}
{ if($1 in spks){ print $2,spks[$1]}}' $list | sort -k2  > $output_path/filtered_list

echo "Creating concatenation lists"
awk -v wd=$input_path/clips -v nc=$num_cat '{ 
wav=wd"/"$1
if(cur_spk!=$2){ 
   cur_spk=$2; 
   num_sess=0; 
   count=0
} 
if(num_sess==0){ 
   count++; 
   sess_id=sprintf("%s-sess%05d",cur_spk,count); 
   file="'$output_path'/lists_cat/"sess_id".txt";
   print sess_id,cur_spk;
}
num_sess++;
print "file",wav > file
if(num_sess==nc) { num_sess=0}
}' $output_path/filtered_list > $output_path/utt2spk

echo "Creating utt2spk, wav.scp, ..."
utils/utt2spk_to_spk2utt.pl $output_path/utt2spk > $output_path/spk2utt

awk '{
file_lst="'$output_path'/lists_cat/"$1".txt";
print $1, "ffmpeg -v 8 -f concat -safe 0 -i "file_lst" -f wav -acodec pcm_s16le - | sox -t wav - -t wav '"$down"' - |";
}' $output_path/utt2spk > $output_path/wav.scp

awk '{ print $1,"'$lang'" }' $output_path/utt2spk > $output_path/utt2lang

utils/fix_data_dir.sh $output_path
utils/validate_data_dir.sh --no-text --no-feats $output_path

num_sess=$(wc -l $output_path/utt2spk | awk '{ print $1}')
num_spks=$(wc -l $output_path/spk2utt | awk '{ print $1}')
echo "Created CommonVoice $lang num_sess=$num_sess num_spks=$num_spks"

