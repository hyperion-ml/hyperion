#!/bin/bash

sre21_eval_root=$1
scores=$2
soft_dir=./sre21/scoring_software/

output_scores=$scores.tsv

trials=$sre21_eval_root/docs/sre21_audio_eval_trials.tsv

awk -v fscores=$scores -f local/format_sre_scores.awk $trials > $output_scores

python3 $soft_dir/sre_submission_validator.py -t audio -o $output_scores \
	-l $trials 

# cp $output_scores $HOME
# cd $HOME
# tar czvf sre20cts_eval_scores.tgz sre20cts_eval_scores.tsv
# cd -
