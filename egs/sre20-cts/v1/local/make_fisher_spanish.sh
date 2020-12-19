#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <fisher-path (LDC2010S01)> <fs 8/16> <OUTPATH>"
    exit 1
fi

input_path=$1
fs=$2
output_path=$3

echo "Preparing Fisher Spanish"
up=""
if [ $fs -eq 16 ];then
    up="sox -t wav - -t wav -e signed-integer -b 16 -r 16k - |"
fi

mkdir -p $output_path

docs=$input_path/fisher_spa_d1/docs
cat $docs/file_d1.tbl $docs/file_d2.tbl | \
    awk -F "," -v fcalls=$docs/fsp06_call.tbl 'BEGIN{
while(getline < fcalls)
{
  spk_a[$1]=$3;
  spk_b[$1]=$9;
}
}
/\.sph$/ {
bn=$1;
sub(/.*\//,"",bn);
sub(/\.sph$/,"",bn);
call_id=$1;
sub(/_fsp\.sph$/,"",call_id);
sub(/.*_/,"",call_id);
spk=spk_a[call_id];
uttid=spk"-"bn"-a"
print uttid,spk,$1,"1";
spk=spk_b[call_id];
uttid=spk"-"bn"-b"
print uttid,spk,$1,"2";
}' | sort -k1,1 > $output_path/table

awk '{ print $1,"sph2pipe -f wav -p -c "$4" '"$input_path"'/"$3" |'"$up"'" }' $output_path/table > $output_path/wav.scp
awk '{ print $1,$2 }' $output_path/table > $output_path/utt2spk
utils/utt2spk_to_spk2utt.pl $output_path/utt2spk > $output_path/spk2utt
awk '{ print $1,"SPA" }' $output_path/table > $output_path/utt2lang

utils/fix_data_dir.sh $output_path
utils/validate_data_dir.sh --no-text --no-feats $output_path
