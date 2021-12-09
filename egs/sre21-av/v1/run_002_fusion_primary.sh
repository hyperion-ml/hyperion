#!/bin/bash
# Copyright
#                2021   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1

echo "This an example of how to fuse audio and visual systems"

fus_name=v1_closed_av_primary
audio=../../sre21-av-a/v1.16k/exp/fusion/v2.5.1_fus_pfus0.1_l21e-3_pcal0.05_l21e-4/3
#visual=../../sre21-av-v/v0.2/exp/fusion/v2.3_ptrn0.05_l21e-4/1
visual=../../sre21-av-v/v0.2/exp/fusion/v2.4_ptrn0.05_l21e-4/2

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
  local/score_sre21_official.sh $sre21_dev_root audio-visual dev $output_dir
    
fi

if [ $stage -le 2 ];then
  # Prepare SRE21 eval
  hyp_utils/conda_env.sh \
    local/sum_fusion.py \
    --ndx-file data/sre21_audio-visual_eval/trials.csv \
    --audio-scores $audio/sre21_audio-visual_eval_scores \
    --visual-scores $visual/sre21_audio-visual_eval_scores \
    --output-scores $output_dir/sre21_audio-visual_eval_scores 

  local/score_sre21.sh data/sre21_audio-visual_eval audio-visual_eval $output_dir
  local/score_sre21_official.sh $sre21_eval_root audio-visual eval $output_dir
  #./local/make_sre21_sub.sh $sre21_eval_root $output_dir/sre21_audio-visual_eval_scores 
    
fi


