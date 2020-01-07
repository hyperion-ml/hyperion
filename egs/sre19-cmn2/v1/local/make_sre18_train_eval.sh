#!/bin/bash

# Copyright 2018 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SRE18_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
segm_file=$docs/sre18_eval_segment_key.tsv

tel_up=""
vid_down=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
    vid_down=" -r 16k "
elif [ $fs -eq 8 ];then
    vid_down=" -r 8k "
fi


#Eval CMN2
output_dir=$output_path/sre18_train_eval_cmn2
mkdir -p $output_dir
awk '$7=="cmn2" && $4 != "unlabeled" { print $2"-"$1,$2}' $segm_file | sort -k1,1 > $output_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt

find $input_path -name "*.sph" > $output_dir/wav.scp.tmp

awk -v fwav=$output_dir/wav.scp.tmp 'BEGIN{
while(getline < fwav)
{
   bn=$1; 
   sub(/.*\//,"",bn);
   wav[bn]=$1;
}
}
$7=="cmn2" && $4 != "unlabeled" {  print $2"-"$1,"sph2pipe -f wav -p -c 1 "wav[$1]" |'"$tel_up"'"}' $segm_file | \
    sort -k1,1 > $output_dir/wav.scp

rm -f $output_dir/wav.scp.tmp

awk -v sf=$segm_file 'BEGIN{
while(getline < sf)
{
 gender[$1]=substr($3,1,1)
}
}
{ sub(/^[^-]*-/,"",$2); print $1,gender[$2] } ' $output_dir/spk2utt > $output_dir/spk2gender

utils/fix_data_dir.sh $output_dir
utils/validate_data_dir.sh --no-text --no-feats $output_dir


#Eval VAST
output_dir=$output_path/sre18_train_eval_vast
mkdir -p $output_dir
awk '$7=="vast" { print $2"-"$1,$2}' $segm_file | sort -k1,1 > $output_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt

find $input_path -name "*.flac" > $output_dir/wav.scp.tmp

awk -v fwav=$output_dir/wav.scp.tmp 'BEGIN{
while(getline < fwav)
{
   bn=$1; 
   sub(/.*\//,"",bn);
   wav[bn]=$1;
}
}
$7=="vast" {  print $2"-"$1,"sox "wav[$1]" -t wav -b 16 -e signed-integer'"$vid_down"' - |"}' $segm_file | \
    sort -k1,1 > $output_dir/wav.scp

rm -f $output_dir/wav.scp.tmp

utils/fix_data_dir.sh $output_dir
utils/validate_data_dir.sh --no-text --no-feats $output_dir


