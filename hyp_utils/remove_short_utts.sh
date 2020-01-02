#!/bin/bash
# Copyright 2017 Johns Hopkins University (David Snyder)
#           2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.

#Removes utterances with less than min_len frames

min_len=400

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <data-dir>"
  echo "e.g.: $0 --min-len 400 data/train_no_sil"
  echo "Options: "
  echo "  --min-len <num-frames>   # minimum number of frames"
  exit 1;
fi

data_dir=$1

mv $data_dir/utt2num_frames $data_dir/utt2num_frames.bak
awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data_dir/utt2num_frames.bak > $data_dir/utt2num_frames
utils/filter_scp.pl $data_dir/utt2num_frames $data_dir/utt2spk > $data_dir/utt2spk.new
mv $data_dir/utt2spk.new $data_dir/utt2spk
utils/fix_data_dir.sh $data_dir


