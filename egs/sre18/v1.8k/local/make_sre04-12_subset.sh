#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
channel=tel
style=phonecall
dur=short
orig_fs=8

. parse_options.sh || exit 1;


if [ $# -ne 5 ]; then
  echo "Usage: $0 [--orig-fs <8/16>] [--channel <tel/mic>] [--style <phonecall,interview>] [--dur <short/long> ] <wav-root> <year> <master-key> <f_sample> <data-dir>"
  echo "e.g.: $0 /export/corpora/LDC/LDC2006S44 04 master_key/NIST_SRE_segments_key.csv 8 data/sre04"
  exit 1;
fi


set -e

wav_root=$1
year=$2
master_key=$3
fs=$4
data_dir=$5

mkdir -p $data_dir


#Find sph files in wav_root
find -L $wav_root -name "*.sph" > $data_dir/wav_list
#find -L $wav_root -name "*.wav" >> $data_dir/wav_list

#Make simplified key
# wav basename channel spk_id gender lang_id
awk -v year=$year -v wl=$data_dir/wav_list '
   BEGIN{
     while(getline < wl)
     {
        bn=$1;
        sub(/.*\//,"",bn);
        wav[bn]=$1;
     }
     FS=",";
   }
   $34==year && $14=="'$channel'" && $15=="'$style'" && $25=="'$dur'" && $41=="N" && $5!="N/A" && $10 != "N/A" && $0 !~ /unacceptable/ { 
     bn=$1;
     sub(/.*\//,"",bn);
     if($3=="x"){ $3="a"}; 
     key=$10"-sre"year"-"$2"-"$3; 
     if (bn in wav){
           print wav[bn], key, $3, $10, $5, $7, $11, $12, $14, $15, $18, $23, $28, $29 ; 
     }
   }' $master_key > $data_dir/key


updownsample_str=""
if [[ "$orig_fs" == "8" ]] && [[ "$fs" == "16" ]];then
    updownsample_str=" sox -t wav - -t wav -r 16k - |"
elif [[ "$orig_fs" == "16" ]] && [[ "$fs" == "8" ]];then
     updownsample_str=" sox -t wav - -t wav -r 8k - |"
fi
awk '{ if( $3=="a" || $3=="x" ){c=1}else{c=2}; 
       print $2" sph2pipe -f wav -p -c "c,$1" |'"$updownsample_str"'" }' \
    $data_dir/key | sort -k1,1 -u > $data_dir/wav.scp

awk '{ print $2" "$4 }' \
    $data_dir/key | sort -k1,1 -u > $data_dir/utt2spk

utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt

awk '{ print $4" "$5 }' \
    $data_dir/key | sort -k1,1 -u > $data_dir/spk2gender

awk '{ print $2" "$7 }' \
    $data_dir/key | sort -k1,1 -u > $data_dir/utt2lang

awk '{ for(i=2;i<NF;i++){ printf "%s ",$i }; printf "%s\n",$NF}' $data_dir/key | sort -k1,1 -u > $data_dir/utt2info

utils/fix_data_dir.sh $data_dir
utils/validate_data_dir.sh --no-text --no-feats $data_dir
