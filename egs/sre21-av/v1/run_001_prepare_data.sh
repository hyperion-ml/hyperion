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

. parse_options.sh || exit 1;
. datapath.sh 

if [ $stage -le 1 ];then
  # Prepare SRE21 dev
  hyp_utils/conda_env.sh \
    local/prepare_sre21av_dev.py \
    --corpus-dir $sre21_dev_root \
    --output-path data/sre21_audio-visual_dev
  
fi

if [ $stage -le 2 ];then
  # Prepare SRE21 eval
  hyp_utils/conda_env.sh \
    local/prepare_sre21av_eval.py \
    --corpus-dir $sre21_eval_root \
    --output-path data/sre21_audio-visual_eval
  
fi

exit
