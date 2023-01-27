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

for cond in o o_clean e e_clean h h_clean
do
    echo "Voxceleb1 $cond"
    key=$data_dir/trials_$cond
    #Compute performance
    python local/score_dcf.py --key-file $key --score-file $score_dir/voxceleb1_scores$suffix --output-path $score_dir/voxceleb1${suffix}_${cond} &
done
wait

