#!/bin/bash

sre20_eval_root=$1
scores=$2
bias=$3
soft_dir=./scoring_software/sre19-cmn2


if [ -z "$bias" ];then
    bias=0
fi

output_scores=$scores.tsv

trials=$sre20_eval_root/docs/2020_cts_challenge_trials.tsv

awk -v fscores=$scores -v bias=$bias -f local/format_sre18_scores.awk $trials > $output_scores

python3 $soft_dir/sre18_submission_validator.py -o $output_scores \
	-l $trials 

cp $output_scores $HOME
cd $HOME
tar czvf sre20cts_eval_scores.tgz sre20cts_eval_scores.tsv
cd -
