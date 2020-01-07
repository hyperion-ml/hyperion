#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-root> <dev/eval> <score-dir>"
  exit 1;
fi

set -e

data_dir=$1
dev_eval=$2
score_dir=$3

# SITW Trials
trial_dir=$data_dir/trials

for cond in core-core core-multi assist-core assist-multi
do
    echo "SITW ${dev_eval} $cond"
    key=$trial_dir/$cond.lst

    #Compute performance
    python local/score_dcf.py --key-file $key --score-file $score_dir/sitw_${dev_eval}_${cond}_scores --output-path $score_dir/sitw_${dev_eval}_${cond} &
done
wait



