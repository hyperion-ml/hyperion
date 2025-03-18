#!/bin/bash
# Copyright 2025 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 3 ]; then
  echo "Usage: $0 <track> <dev/eval> <score-dir>"
  exit 1;
fi

set -e
. datapath.sh

track=$1
subset=$2
score_dir=$3

echo "Score SRE24 ${track} ${subset} for $score_dir"

soft_dir=./sre24_scorer

if [ "$subset" == "dev" ];then
  docs_root=$sre24_dev_docs_root/docs
else
  docs_root=$sre24_eval_root/docs
fi

if [ ! -f $soft_dir/sre24_submission_scorer.py ];then
    echo "downloading scoring tool"
    local/download_sre24_scoring_tool.sh
fi


scores=$score_dir/sre24_${track}_${subset}_scores.tsv
results=$score_dir/sre24_${track}_${subset}_official_results.txt

trials=$docs_root/sre24_${track}_${subset}_trials.tsv
key=$docs_root/sre24_${track}_${subset}_trial_key.tsv

python3 $soft_dir/sre_submission_validator.py -t $track -o $scores \
        -l $trials

python3 $soft_dir/sre24_submission_scorer.py -t $track -o $scores \
        -l $trials -r $key | tee $results

