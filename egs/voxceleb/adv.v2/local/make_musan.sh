#!/bin/bash
# Copyright 2015   David Snyder
# Copyright 2019   Johns Hopkins University (Jesus Villalba) (added fs support)
# Apache 2.0.
#
# This script, called by ../run.sh, creates the MUSAN
# data directory. The required dataset is freely available at
#   http://www.openslr.org/17/

set -e
use_vocals='Y'

. parse_options.sh || exit 1;

if [ $# -ne 3 ];then
    echo "Usage: $0 [options] <in-dir> <fs> <data-dir>";
    echo "e.g.: $0 /export/corpora/JHU/musan 8 data"
    exit 1;
fi

in_dir=$1
fs=$2
data_dir=$3

mkdir -p $data_dir/musan.tmp

echo "Preparing ${data_dir}/musan..."
mkdir -p ${data_dir}/musan
local/make_musan.py ${in_dir} $fs ${data_dir}/musan ${use_vocals}

utils/fix_data_dir.sh ${data_dir}/musan

grep "music" ${data_dir}/musan/utt2spk > $data_dir/musan.tmp/utt2spk_music
grep "speech" ${data_dir}/musan/utt2spk > $data_dir/musan.tmp/utt2spk_speech
grep "noise" ${data_dir}/musan/utt2spk > $data_dir/musan.tmp/utt2spk_noise
utils/subset_data_dir.sh --utt-list $data_dir/musan.tmp/utt2spk_music \
  ${data_dir}/musan ${data_dir}/musan_music
utils/subset_data_dir.sh --utt-list $data_dir/musan.tmp/utt2spk_speech \
  ${data_dir}/musan ${data_dir}/musan_speech
utils/subset_data_dir.sh --utt-list $data_dir/musan.tmp/utt2spk_noise \
  ${data_dir}/musan ${data_dir}/musan_noise

utils/fix_data_dir.sh ${data_dir}/musan_music
utils/fix_data_dir.sh ${data_dir}/musan_speech
utils/fix_data_dir.sh ${data_dir}/musan_noise

rm -rf $data_dir/musan.tmp

