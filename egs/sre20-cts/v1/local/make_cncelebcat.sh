#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 4 ]; then
    echo "Usage: $0 <CN-CELEB-PATH> <dev/eval> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
deveval=$2
fs=$3
output_path=$4

tel_down=""
if [ $fs -eq 8 ];then
    tel_down=" -r 8k "
fi


mkdir -p $output_path

spklist=$input_path/dev/dev.lst
find $input_path/data -name "*.wav" | \
    awk -v spklist=$spklist -v de=$deveval 'BEGIN{
while(getline < spklist){ spks["cn-celeb-"$1]=1;}
}
{
wav=$1;
sub(/.*data\//,"",$1);
nf=split($1,f,"/");
spk="cn-celeb-"f[1];
bn=f[2];
sess=spk"-"bn;
sub(/-[^-]*\.wav$/,"",sess);
if(de=="dev"){
  if (spk in spks){ print sess, spk, wav }
}
else{
  if (!(spk in spks)){ print sess, spk, wav }
}
}' > $output_path/table

awk '{ print $1,$2}' $output_path/table | sort -u -k1,1  > $output_path/utt2spk
utils/utt2spk_to_spk2utt.pl $output_path/utt2spk > $output_path/spk2utt

awk '{ print $3, $1}' $output_path/table | sort -k 1,1 | \
    utils/utt2spk_to_spk2utt.pl | \
    awk '{
printf "%s sox",$1; 
for(i=2;i<=NF;i++){ printf " %s",$i };
print " -t wav -b 16 -e signed-integer '"$tel_down"' - |" 
}' > $output_path/wav.scp

utils/fix_data_dir.sh $output_path
utils/validate_data_dir.sh --no-text --no-feats $output_path



