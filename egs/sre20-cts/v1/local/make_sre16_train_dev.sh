#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SRE16_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
meta=$input_path/metadata
call2lang=$meta/calls.tsv
call2spk=$meta/call_sides.tsv
spk2gender=$meta/subjects.tsv
segm_file=$docs/sre16_dev_segment_key.tsv

tel_up=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
fi

#Dev CMN2 Mandarin and Cebuano
for lang in cmn ceb
do
    output_dir=$output_path/sre16_train_dev_$lang
    mkdir -p $output_dir
    awk -v c2l=$call2lang -v c2s=$call2spk -v s2g=$spk2gender -v l=$lang -F "\t" 'BEGIN{ 
while(getline < c2l)
{
     if($2 == l){ calls[$1]=1 }
}
while(getline < c2s) { spk[$1]=$3 }
while(getline < s2g) { gender[$1]=tolower($2) }
}
{ if($2 in calls) { s=spk[$2]; print $1, s, gender[s] }}' $segm_file > $output_dir/table

    awk '{ print $2"-"$1,$2}' $output_dir/table | sort -k1,1 > $output_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt
    awk '{ print $2,$3}' $output_dir/table | sort -k1,1 -u > $output_dir/spk2gender
    awk -v lang=$lang '{ print $1,toupper(lang)}' $output_dir/utt2spk > $output_dir/utt2lang
    
    find -L $input_path -name "*.sph" > $output_dir/wav.scp.tmp    

    awk -v fwav=$output_dir/wav.scp.tmp 'BEGIN{
while(getline < fwav)
{
   bn=$1; 
   sub(/.*\//,"",bn);
   sub(/\.sph$/,"",bn);
   wav[bn]=$1;
}
}
{  print $2"-"$1,"sph2pipe -f wav -p -c 1 "wav[$1]" |'"$tel_up"'"}' $output_dir/table | \
    sort -k1,1 > $output_dir/wav.scp

    rm -f $output_dir/wav.scp.tmp
    utils/fix_data_dir.sh $output_dir
    utils/validate_data_dir.sh --no-text --no-feats $output_dir
done


