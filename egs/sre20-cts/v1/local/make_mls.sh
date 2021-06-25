#!/bin/bash

# Copyright 2021 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 4 ]; then
    echo "Usage: $0 <MLS-PATH> <lang-id> <fs 8/16> <OUTPATH>"
    exit 1
fi

input_path=$1
lang=$2
fs=$3
output_path=$4

echo "Preparing MLS $lang"
down=""
if [ $fs -eq 8 ];then
    down=" -r 8k "
else
    down=""
fi
input_path=$input_path/mls_$lang
mkdir -p $output_path

for t in train dev test
do
    awk '{ spk=$1; sub(/_.*$/,"",spk); 
sub(/.*\//,"",$2); sub(/\.mp3$/,"",$2);
spk="mls-'$lang'-"spk
print $1, spk"-"$2, spk }' $input_path/$t/segments.txt
done > $output_path/file2utt2spk

find $input_path -name "*.flac" | sort | awk '{ bn=$1; sub(/.*\//,"",bn); sub(/\.flac$/,"",bn); print bn,$1 }' > $output_path/file2wav

awk '{ print $2,$3}' $output_path/file2utt2spk | sort -k 1,1 > $output_path/utt2spk
awk '{ print $1,"'$lang'"}' $output_path/utt2spk | sort -k 1,1 > $output_path/utt2lang
utils/utt2spk_to_spk2utt.pl $output_path/utt2spk > $output_path/spk2utt

awk -v f2w=$output_path/file2wav 'BEGIN{
while(getline < f2w)
{
    v[$1]=$2;
}
}
{ w[$2]=w[$2]" "v[$1]; }
END{
for(n in w)
{
   print n,"sox "w[n]" -t wav '"$down"' - |"
}
}' $output_path/file2utt2spk | sort -k1,1 > $output_path/wav.scp

utils/fix_data_dir.sh $output_path
num_sess=$(wc -l $output_path/utt2spk | awk '{ print $1}')
num_spks=$(wc -l $output_path/spk2utt | awk '{ print $1}')
echo "Created MLS $lang num_sess=$num_sess num_spks=$num_spks"




