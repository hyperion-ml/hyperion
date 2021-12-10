#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-root> <subset> <score-dir>"
  exit 1;
fi

set -e

data_dir=$1
subset=$2
score_dir=$3

key=$data_dir/trials

echo "SRE21 ${subset}"

#Compute performance
hyp_utils/conda_env.sh \
  local/score_sre21.py --key-file $key \
  --score-file $score_dir/sre21_${subset}_scores \
  --sre21-subset $subset \
  --output-file $score_dir/sre21_${subset}_results.csv
