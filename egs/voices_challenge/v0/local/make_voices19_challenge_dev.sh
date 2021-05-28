#!/bin/bash
# Copyright 2019  Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

if [  $# != 3 ]; then
    echo "$0 <db-path> <list-path> <output_path>"
    exit 1
fi
input_path=$1
list_path=$2
output_path=$3

audio_path=$input_path/Development_Data/Speaker_Recognition

echo "$0 making voices challenge dev enroll"
data_out=$output_path/voices19_challenge_dev_enroll
mkdir -p $data_out

enroll_list=$list_path/dev-enroll.lst

awk '{ split($2,f,"/"); spk=f[2]; 
       print spk"-"$2,"'$audio_path'/"$2 }' \
    $enroll_list | sort -k1,1 > $data_out/wav.scp

awk '{ split($2,f,"/"); spk=f[2]; 
       print spk"-"$2,spk}' \
    $enroll_list | sort -k1,1 > $data_out/utt2spk

utils/utt2spk_to_spk2utt.pl $data_out/utt2spk > $data_out/spk2utt

awk '{ split($2,f,"/"); spk=f[2]; 
       print spk"-"$2,$1}' \
    $enroll_list | sort -k1,1 > $data_out/utt2model

utils/utt2spk_to_spk2utt.pl $data_out/utt2model > $data_out/model2utt

awk '{ utt=$2;
       sub(/.*VOiCES-/,"",$2); 
       sub(/\.wav$/,"",$2); 
       split($2,f,"-"); spk=f[3]; 
       utt=spk"-"utt;
       printf "%s %s %s %s", utt,spk,f[1],f[2];
       for(i=4;i<=9;i++){ printf " %s", f[i] };
       printf "\n" }' \
    $enroll_list | sort -k1,1 > $data_out/utt2info

utils/fix_data_dir.sh --utt_extra_files utt2info $data_out

######

echo "$0 making voices challenge dev test"
data_out=$output_path/voices19_challenge_dev_test
mkdir -p $data_out

test_list=$list_path/dev-test.lst

awk '{ split($1,f,"/"); spk=f[2]; 
       print spk"-"$1,"'$audio_path'/"$1 }' \
    $test_list | sort -k1,1 > $data_out/wav.scp

awk '{ split($1,f,"/"); spk=f[2]; 
       print spk"-"$1,spk}' \
    $test_list | sort -k1,1 > $data_out/utt2spk

utils/utt2spk_to_spk2utt.pl $data_out/utt2spk > $data_out/spk2utt


awk '{ utt=$1;
       sub(/.*VOiCES-/,"",$1); 
       sub(/\.wav$/,"",$1); 
       split($1,f,"-"); spk=f[3]; 
       utt=spk"-"utt;
       printf "%s %s %s %s", utt,spk,f[1],f[2];
       for(i=4;i<=9;i++){ printf " %s", f[i] };
       printf "\n" }' \
    $test_list | sort -k1,1 > $data_out/utt2info


key=$list_path/dev-trial-keys.lst

awk '{ split($2,f,"/"); spk=f[2]; 
       sub(/imp/,"nontarget",$3); 
       sub(/tgt/,"target",$3);
       print $1,spk"-"$2,$3}' \
    $key | sort -k1,1 > $data_out/trials


utils/fix_data_dir.sh --utt_extra_files utt2info $data_out



