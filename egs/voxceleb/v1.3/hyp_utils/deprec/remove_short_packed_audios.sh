#!/bin/bash
#           2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.

# Removes short audios using based on the number of samples in packed_audio.scp

fs=16000
min_len=4

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <data-dir>"
  echo "e.g.: $0 --min-len 4 data/train_no_sil"
  echo "Options: "
  echo "  --min-len <num-secs|4>   # minimum number of secods"
  echo "  --fs <fs|1600s>   # sample freq"
  exit 1;
fi

data_dir=$1

awk -v min_len=${min_len} -v fs=$fs '{ 
    sub(/.*\[/,"",$2); 
    sub(/\]$/,"",$2); 
    nf=split($2,f,":");
    t=(f[2]-f[1]+1)/fs;
    if (t>=min_len) { print $1,t }}' \
	$data_dir/packed_audio.scp > $data_dir/utt2durs
utils/filter_scp.pl $data_dir/utt2durs $data_dir/utt2spk > $data_dir/utt2spk.new
mv $data_dir/utt2spk.new $data_dir/utt2spk
utils/fix_data_dir.sh --utt-extra-files packed_audio.scp $data_dir


