#!/bin/bash

sre18_eval_root=$1
cmn2_scores=$2
vast_scores=$3
output_dir=$4
soft_dir=./scoring_software_sre18

mkdir -p $output_dir
scores=$output_dir/sre18_eval_scores.tsv

trials=$sre18_eval_root/docs/sre18_eval_trials.tsv

cat $cmn2_scores $vast_scores | sort > $scores.tmp


awk -v fscores=$scores.tmp -f local/format_sre18_scores.awk $trials > $scores

python3 $soft_dir/sre18_submission_validator.py -o $scores \
	-l $trials 

   

