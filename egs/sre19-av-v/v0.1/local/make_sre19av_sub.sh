#!/bin/bash

sre19_eval_root=$1
scores=$2
soft_dir=../../sre19-cmn2/v1/scoring_software/sre19-cmn2

output_scores=$scores.tsv

trials=$sre19_eval_root/docs/sre19_av_eval_trials.tsv

awk -v fscores=$scores -f local/format_sre18_scores.awk $trials > $output_scores

python3 $soft_dir/sre18_submission_validator.py -o $output_scores \
	-l $trials 

   

