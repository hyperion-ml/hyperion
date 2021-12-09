#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 4 ]; then
  echo "Usage: $0 <sre21-dev/eval-root> <track> <dev/eval> <score-dir>"
  exit 1;
fi

set -e

sre21_root=$1
track=$2
subset=$3
score_dir=$4

echo "Score SRE21 ${track} ${subset} for $score_dir"

soft_dir=./sre21/scoring_software

if [ ! -f $s_dir/sre_scorer.py ];then
    echo "downloading scoring tool"
    local/download_sre21_scoring_tool.sh
fi


scores=$score_dir/sre21_${track}_${subset}_scores
results=$score_dir/sre21_${track}_${subset}_official_results

trials=$sre21_root/docs/sre21_${track}_${subset}_trials.tsv
key=$sre21_root/docs/sre21_${track}_${subset}_trial_key.tsv

awk -v fscores=$scores -f local/format_sre_scores.awk $trials > $scores.tsv

python3 $soft_dir/sre_submission_validator.py -t $track -o $scores.tsv \
	-l $trials

python3 $soft_dir/sre21_submission_scorer.py -t $track -o $scores.tsv \
	-l $trials -r $key | tee $results
rm $scores.tsv
