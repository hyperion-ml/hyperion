#!/bin/bash
# Copyright 2018   Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Links Mixer-6 mic-0x signals to lavalier mic-02 
data_dir=$1

awk -v k=$data_dir/utt2spk 'BEGIN{
while(getline < k){
    ch=$1;
    sub(/.*_/,"",ch);
    if(ch == "02"){
        name=$1;
        sub(/_[^_]*$/,"",name);
        clean[name] = $1
    }
}
}
{ name=$1;
  sub(/_[^_]*$/,"",name);
  if (name in clean) { c=clean[name] } else { c="N/A" };
  print $1" "c }' $data_dir/utt2spk > $data_dir/utt2clean

