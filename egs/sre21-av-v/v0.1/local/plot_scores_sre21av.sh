#!/bin/bash

score_dir=$1
output_dir=$2
system_name=$3
mkdir -p $output_dir

key_sre21a_dev=data/sre21_visual_dev_test/trials
key_sre21a_eval=data/sre21_visual_eval_test/trials
key_sre21av_dev=data/sre21_visual_dev_test/trials
key_sre21av_eval=data/sre21_visual_eval_test/trials_av

local/plot_scores_sre21av.py \
    --key-sre21a-dev $key_sre21a_dev \
    --scores-sre21a-dev $score_dir/sre21_visual_dev_scores \
    --key-sre21a-eval $key_sre21a_eval \
    --scores-sre21a-eval $score_dir/sre21_visual_eval_scores \
    --key-sre21av-dev $key_sre21av_dev \
    --scores-sre21av-dev $score_dir/sre21_visual_dev_scores \
    --key-sre21av-eval $key_sre21av_eval \
    --scores-sre21av-eval $score_dir/sre21_audio-visual_eval_scores \
    --output-path $output_dir --name "$system_name"




