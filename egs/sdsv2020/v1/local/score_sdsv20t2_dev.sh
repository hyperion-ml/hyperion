#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-root> <score-dir>"
  exit 1;
fi

set -e

data_dir=$1
score_dir=$2

echo "Score SDSV20 dev Task2"
key=$data_dir/trials
#Compute performance
python local/score_dcf.py --key-file $key --score-file $score_dir/sdsv20t2_dev_scores --output-path $score_dir/sdsv20t2_dev


