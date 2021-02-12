#!/bin/bash
# Copyright 2017 Johns Hopkins University (David Snyder)
#           2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.

#Removes speakers with few utterances

min_num_utts=8

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

kaldi_utils=hyp_utils/kaldi/utils

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <data-dir>"
  echo "e.g.: $0 --min-num-utt 8 data/train_no_sil"
  echo "Options: "
  echo "  --min-num-utts <num-utts>  # minimum number utterances per speaker"
  exit 1;
fi

data_dir=$1

awk '{print $1, NF-1}' $data_dir/spk2utt > $data_dir/spk2num
awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data_dir/spk2num | ${kaldi_utils}/filter_scp.pl - $data_dir/spk2utt > $data_dir/spk2utt.new
mv $data_dir/spk2utt.new $data_dir/spk2utt
${kaldi_utils}/spk2utt_to_utt2spk.pl $data_dir/spk2utt > $data_dir/utt2spk

utt_extra=""
if [ -f $data_dir/utt2num_frames ];then
    utt_extra=utt2num_frames
fi
if [ -f $data_dir/packed_audio.scp ];then
    utt_extra="${utt_extra} packed_audio.scp"
fi

#${kaldi_utils}/filter_scp.pl $data_dir/utt2spk $data_dir/utt2num_frames > $data_dir/utt2num_frames.new
#mv $data_dir/utt2num_frames.new $data_dir/utt2num_frames

# Now we're ready to create training examples.
if [ -n "${utt_extra}" ];then
    ${kaldi_utils}/fix_data_dir.sh --utt-extra-files "$utt_extra" $data_dir
else
    ${kaldi_utils}/fix_data_dir.sh $data_dir
fi
