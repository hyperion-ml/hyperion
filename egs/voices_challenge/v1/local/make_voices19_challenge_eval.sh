#!/bin/bash
# Copyright 2019  Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

if [  $# != 4 ]; then
    echo "$0 <db-path> <list-path> <key-path> <output_path>"
    exit 1
fi
input_path=$1
list_path=$2
key_path=$3
output_path=$4

map_file=$key_path/VOiCES_challenge_2019_eval.SID.map
key=$key_path/VOiCES_challenge_2019_eval.SID.trial-keys.lst

audio_path=$input_path/Evaluation_Data/Speaker_Recognition

echo "$0 making voices challenge eval enroll"
data_out=$output_path/voices19_challenge_eval_enroll
mkdir -p $data_out

enroll_list=$list_path/eval-enroll.lst

awk '{ print $2,"'$audio_path'/"$2 }' \
    $enroll_list | sort -k1,1 > $data_out/wav.scp

awk '{ print $2,$2}' \
    $enroll_list | sort -k1,1 > $data_out/utt2spk

utils/utt2spk_to_spk2utt.pl $data_out/utt2spk > $data_out/spk2utt

awk '{ print $2,$1}' \
    $enroll_list | sort -k1,1 > $data_out/utt2model

utils/utt2spk_to_spk2utt.pl $data_out/utt2model > $data_out/model2utt


awk -v fmap=$map_file 'BEGIN{
while(getline < fmap)
{
    sub(/.*VOiCES-/,"",$2); 
    sub(/\.wav$/,"",$2); 
    map[$1]=$2
}
}
{      utt=$2;
       sub(/sid_eval\//,"",$2); 
       sub(/\.wav$/,"",$2); 
       orig_utt=map[$2];
       split(orig_utt,f,"-"); 
       if(f[1]=="src"){
           spk=f[2];        
           printf "%s %s %s none %s %s", utt,spk,f[1],f[3],f[4];
           for(i=0;i<=4;i++){ printf " N/A" };
       }
       else {
           spk=f[3];
           printf "%s %s %s %s", utt,spk,f[1],f[2];
           for(i=4;i<=9;i++){ printf " %s", f[i] };
       }
       printf "\n" }' \
    $enroll_list | sort -k1,1 > $data_out/utt2info



utils/fix_data_dir.sh --utt_extra_files utt2info $data_out
#utils/fix_data_dir.sh $data_out

######

echo "$0 making voices challenge eval test"
data_out=$output_path/voices19_challenge_eval_test
mkdir -p $data_out

test_list=$list_path/eval-test.lst

awk '{ print $1,"'$audio_path'/"$1 }' \
    $test_list | sort -k1,1 > $data_out/wav.scp

awk '{ print $1,$1}' \
    $test_list | sort -k1,1 > $data_out/utt2spk

utils/utt2spk_to_spk2utt.pl $data_out/utt2spk > $data_out/spk2utt

awk -v fmap=$map_file 'BEGIN{
while(getline < fmap)
{
    sub(/.*VOiCES-/,"",$2); 
    sub(/\.wav$/,"",$2); 
    map[$1]=$2
}
}
{      utt=$1;
       sub(/sid_eval\//,"",$1); 
       sub(/\.wav$/,"",$1); 
       orig_utt=map[$1];
       split(orig_utt,f,"-"); 
       if(f[1]=="src"){
           spk=f[2];        
           printf "%s %s %s none %s %s", utt,spk,f[1],f[3],f[4];
           for(i=0;i<=4;i++){ printf " N/A" };
       }
       else {
           spk=f[3];
           printf "%s %s %s %s", utt,spk,f[1],f[2];
           for(i=4;i<=9;i++){ printf " %s", f[i] };
       }
       printf "\n" }' \
    $test_list | sort -k1,1 > $data_out/utt2info

# awk '{ utt=$1;
#        sub(/.*VOiCES-/,"",$1); 
#        sub(/\.wav$/,"",$1); 
#        split($1,f,"-"); spk=f[3]; 
#        utt=spk"-"utt;
#        printf "%s %s %s %s", utt,spk,f[1],f[2];
#        for(i=4;i<=9;i++){ printf " %s", f[i] };
#        printf "\n" }' \
#     $test_list | sort -k1,1 > $data_out/utt2info


awk '{ split($2,f,"/"); spk=f[2]; 
       sub(/imp/,"nontarget",$3); 
       sub(/tgt/,"target",$3);
       print $1,"sid_eval/"$2,$3}' \
    $key | sort -k1,1 > $data_out/trials

#key=$list_path/eval-trial.lst

#awk '{ print $1,$2}' \
#    $key | sort -k1,1 > $data_out/trials


utils/fix_data_dir.sh --utt_extra_files utt2info $data_out
#utils/fix_data_dir.sh $data_out



