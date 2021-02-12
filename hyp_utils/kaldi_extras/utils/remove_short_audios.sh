#!/bin/bash
#           2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.

# Removes short audios using based on the utt2dur

min_len=4

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

kaldi_utils=hyp_utils/kaldi/utils

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <data-dir>"
  echo "e.g.: $0 --min-len 4 data/train_no_sil"
  echo "Options: "
  echo "  --min-len <num-secs|4>   # minimum number of secods"
  exit 1;
fi

data_dir=$1

awk -v min_len=${min_len} '{ 
    if ($2>=min_len) { print $1,t }}' \
	$data_dir/utt2dur > $data_dir/utt2dur.new
${kaldi_utils}/filter_scp.pl $data_dir/utt2dur.new $data_dir/utt2spk > $data_dir/utt2spk.new
mv $data_dir/utt2spk.new $data_dir/utt2spk
${kaldi_utils}/fix_data_dir.sh --utt-extra-files utt2dur $data_dir


