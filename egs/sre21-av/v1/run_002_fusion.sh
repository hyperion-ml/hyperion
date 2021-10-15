#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

# config_file=default_config.sh
stage=1

fus_name=v1
audio=../../sre21-av-a/v1.16k/exp/fusion/v2.2.1_fus_pfus0.1_l21e-3_pcal0.05_l21e-4/3
visual=../../sre21-av-v/v0.2/exp/fusion/v1.1_ptrn0.1_l21e-4/0

. parse_options.sh || exit 1;
. datapath.sh

output_dir=exp/fusion/$fus_name
mkdir -p $output_dir

if [ $stage -le 1 ];then
  # Prepare SRE21 dev
  hyp_utils/conda_env.sh \
    local/sum_fusion.py \
    --ndx-file data/sre21_audio-visual_dev/trials.csv \
    --audio-scores $audio/sre21_audio-visual_dev_scores \
    --visual-scores $visual/sre21_visual_dev_scores \
    --output-scores $output_dir/sre21_audio-visual_dev_scores

  local/score_sre21.sh data/sre21_audio-visual_dev audio-visual_dev $output_dir

fi

if [ $stage -le 1 ];then
  # Prepare SRE21 eval
  hyp_utils/conda_env.sh \
    local/sum_fusion.py \
    --ndx-file data/sre21_audio-visual_eval/trials.csv \
    --audio-scores $audio/sre21_audio-visual_eval_scores \
    --visual-scores $visual/sre21_visual_eval_scores \
    --output-scores $output_dir/sre21_audio-visual_eval_scores \
    
fi


