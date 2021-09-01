#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-root> <score-dir>"
  exit 1;
fi

set -e

data_dir=$1
score_dir=$2

key=$data_dir/trials

echo "SRE CTS Superset Dev"

#Compute performance
hyp_utils/conda_env.sh \
  local/score_sre_cts_superset.py --key-file $key \
  --score-file $score_dir/sre_cts_superset_dev_scores \
  --output-file $score_dir/sre_cts_superset_dev_results.csv \
