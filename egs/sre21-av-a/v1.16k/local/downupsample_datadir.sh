#!/bin/bash
# Copyright 2021 Johns Hopkins University (Author Jesus Villalba)
# Apache 2.0

# Creates a new data directory where the wav.scp
# downasmples widebad data to 8k and downsamples back to 16k

if (($# != 3 && $# != 2)); then
  echo "Usage: $0 <input_path> <output_path> [<suffix>]"
  exit 1
fi
input_path=$1
output_path=$2
suffix=$3

mkdir -p $output_path
for f in feats.scp vad.scp utt2spk utt2nation utt2dur utt2speech_dur utt2num_frames utt2lang utt2phone utt2session utt2model text
do
  if [ -f $input_path/$f ];then
    if [ -z "$suffix" ];then
      cp $input_path/$f $output_path/$f
    else
      awk '{ $1=$1"-'$suffix'"; print $0 }' \
	  $input_path/$f > $output_path/$f
    fi
  fi
done

for f in spk2gender
do
  if [ -f $input_path/$f ];then
      cp $input_path/$f $output_path/$f
  fi
done

awk -v p="$suffix" '{
if(p != "") { $1=$1"-"p; }
if(NF==2) {
  print $1, "sox $2 -r 8000 -e signed-integer -b 16 -t raw - | sox -r 8000 -e signed-integer -b 16 -t raw - -r 16000 -t wav - |";
}
else{
  print $0,"sox -t wav - -r 8000 -e signed-integer -b 16 -t raw - | sox -r 8000 -e signed-integer -b 16 -t raw - -r 16000 -t wav - |";
}
}' $input_path/wav.scp > $output_path/wav.scp

utils/utt2spk_to_spk2utt.pl $output_path/utt2spk > $output_path/spk2utt

utils/fix_data_dir.sh $output_path
utils/validate_data_dir.sh --no-text --no-feats $output_path


