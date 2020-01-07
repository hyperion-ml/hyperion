#!/bin/bash

score_dir=$1
output_dir=$2
system_name=$3
mkdir -p $output_dir

key_sre18_dev=data/sre18_dev_test_cmn2/trials
key_sre18_eval=data/sre18_eval_test_cmn2/trials
key_sre19=data/sre19_eval_test_cmn2/trials

local/plot_scores_sre19cmn2.py \
    --key-sre18-dev $key_sre18_dev \
    --scores-sre18-dev $score_dir/sre18_dev_cmn2_scores \
    --key-sre18-eval $key_sre18_eval \
    --scores-sre18-eval $score_dir/sre18_eval_cmn2_scores \
    --key-sre19-eval $key_sre19 \
    --scores-sre19-eval $score_dir/sre19_eval_cmn2_scores \
    --output-path $output_dir --name "$system_name"




