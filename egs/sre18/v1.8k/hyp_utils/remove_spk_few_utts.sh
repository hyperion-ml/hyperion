#!/bin/bash
# Copyright 2017 Johns Hopkins University (David Snyder)
#           2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.

#Removes speakers with few utterances

min_num_utts=8

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <data-dir>"
  echo "e.g.: $0 --min-num-utt 8 data/train_no_sil"
  echo "Options: "
  echo "  --min-num-utts <num-utts>  # minimum number utterances per speaker"
  exit 1;
fi

data_dir=$1

awk '{print $1, NF-1}' data/$data_dir/spk2utt > data/$data_dir/spk2num
awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/$data_dir/spk2num | utils/filter_scp.pl - data/$data_dir/spk2utt > data/$data_dir/spk2utt.new
mv data/$data_dir/spk2utt.new data/$data_dir/spk2utt
utils/spk2utt_to_utt2spk.pl data/$data_dir/spk2utt > data/$data_dir/utt2spk

utils/filter_scp.pl data/$data_dir/utt2spk data/$data_dir/utt2num_frames > data/$data_dir/utt2num_frames.new
mv data/$data_dir/utt2num_frames.new data/$data_dir/utt2num_frames

# Now we're ready to create training examples.
utils/fix_data_dir.sh data/$data_dir
