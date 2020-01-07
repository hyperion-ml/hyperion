#!/bin/bash

score_dir=$1
output_dir=$2
system_name=$3
mkdir -p $output_dir

key_sre18_eval_vast=data/sre18_eval_test_vast/trials
key_sre19_dev=data/sre19_av_a_dev_test/trials
key_sre19_eval=data/sre19_av_a_eval_test/trials

local/plot_scores_sre19av_sre18-9.py \
    --key-sre18-eval $key_sre18_eval_vast \
    --scores-sre18-eval $score_dir/sre18_eval_vast_scores \
    --key-sre19-dev $key_sre19_dev \
    --scores-sre19-dev $score_dir/sre19_av_a_dev_scores \
    --key-sre19-eval $key_sre19_eval \
    --scores-sre19-eval $score_dir/sre19_av_a_eval_scores \
    --output-path $output_dir --name "$system_name"




