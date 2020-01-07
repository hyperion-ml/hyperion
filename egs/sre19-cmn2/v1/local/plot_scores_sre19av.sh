#!/bin/bash

score_dir=$1
output_dir=$2
system_name=$3
mkdir -p $output_dir

key_sitw_core=data/sitw_eval_test/trials/core-core.lst
key_sitw_core_multi=data/sitw_eval_test/trials/core-multi.lst
key_sre18_eval_vast=data/sre18_eval_test_vast/trials
key_sre19_dev=data/sre19_av_a_dev_test/trials
key_sre19_eval=data/sre19_av_a_eval_test/trials
key_janus_dev=data/janus_dev_test_core/trials
key_janus_eval=data/janus_eval_test_core/trials

local/plot_scores_sre19av.py \
    --key-sitw-core $key_sitw_core \
    --scores-sitw-core $score_dir/sitw_eval_core-core_scores \
    --key-sitw-core-multi $key_sitw_core_multi \
    --scores-sitw-core-multi $score_dir/sitw_eval_core-multi_scores \
    --key-sre18-eval $key_sre18_eval_vast \
    --scores-sre18-eval $score_dir/sre18_eval_vast_scores \
    --key-sre19-eval $key_sre19_eval \
    --scores-sre19-eval $score_dir/sre19_av_a_eval_scores \
    --key-sre19-dev $key_sre19_dev \
    --scores-sre19-dev $score_dir/sre19_av_a_dev_scores \
    --key-sre19-eval $key_sre19_eval \
    --scores-sre19-eval $score_dir/sre19_av_a_eval_scores \
    --key-janus-dev $key_janus_dev \
    --scores-janus-dev $score_dir/janus_dev_core_scores \
    --key-janus-eval $key_janus_eval \
    --scores-janus-eval $score_dir/janus_eval_core_scores \
    --output-path $output_dir --name "$system_name"




