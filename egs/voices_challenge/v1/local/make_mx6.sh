#!/bin/bash
# Copyright 2019   Johns Hopkins University (Jesus Villalba)
# Copyright 2017   Johns Hopkins University (David Snyder)
# Apache 2.0.
#
# This script prepares both the microphone and telephone portions of the
# Mixer 6 corpus.
if [ $# -ne 3 ]; then
  echo "Usage: $0 <mixer6-speech> <f_sample> <out-dir>"
  echo "e.g.: $0 /export/corpora/LDC/LDC2013S03 16 data/"
  exit 1;
fi

set -e
in_dir=$1
fs=$2
out_dir=$3

# Mic 01 is the lapel mic for the interviewer, so we don't use it.  Mic 02 is
# the lapel mic for the interviewee.  All other mics are placed throughout the
# room.  In addition to mic 01, we omit mics 03 and 14 as they are often
# silent.
echo "$0: preparing mic speech (excluding 01, 03, and 14)"

for mic in 02 04 05 06 07 08 09 10 11 12 13; do
  local/make_mx6_mic.pl $in_dir $mic $fs $out_dir
done

utils/combine_data.sh --extra-files "utt2clean utt2info" $out_dir/mx6_mic_04_to_13 $out_dir/mx6_mic_{04,05,06,07,08,09,10,11,12,13}

# Mics 02-13 contain the same content, but recorded from different microphones.
# To get some channel diversity, but not be overwhelmed with duplicated data
# we take a 2k subset from mics 04-13 and combine it with all of mic 02.
echo "$0: selecting a 2k subset of mics 04 through 13 and combining it with mic 02"
utils/subset_data_dir.sh $out_dir/mx6_mic_04_to_13 2000 $out_dir/mx6_mic_04_to_13_2k
for f in utt2clean utt2info
do
  cp $out_dir/mx6_mic_04_to_13/$f $out_dir/mx6_mic_04_to_13_2k/$f
done
utils/fix_data_dir.sh --utt-extra-files "utt2clean utt2info" $out_dir/mx6_mic_04_to_13_2k
utils/combine_data.sh --extra-files "utt2clean utt2info" $out_dir/mx6_mic $out_dir/mx6_mic_02 $out_dir/mx6_mic_04_to_13_2k

#echo "$0 make utt2clean list linking mic-0x to mic-02"
#local/make_mx6_utt2clean.sh $out_dir/mx6_mic

echo "$0: preparing telephone portion"
local/make_mx6_calls.pl $in_dir $fs $out_dir

echo "$0 combining mic and telephone speech in data/mx6"
utils/combine_data.sh --extra-files "utt2clean utt2info" $out_dir/mx6 $out_dir/mx6_mic $out_dir/mx6_calls
utils/fix_data_dir.sh --utt-extra-files "utt2clean utt2info" $out_dir/mx6

