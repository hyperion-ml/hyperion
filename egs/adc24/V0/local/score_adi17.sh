#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 2 ] && [ $# -n 3]; then
  echo "Usage: $0 <data-root> <score-dir> [suffix]"
  exit 1;
fi

set -e

data_dir=$1
score_dir=$2
suffix=$3


echo "test adi17"
key=$data_dir/trials
#Compute performance
python local/lscore_dcf.py --key-file $key --score-file $score_dir/adi17_scores$suffix --output-path $score_dir/adi17${suffix} &

wait

