#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1

config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 

if [ $stage -le 1 ];then
  # # Prepare sre19
  # local/make_sre19av_v_dev.sh $sre19_dev_root data
  # local/make_sre19av_v_eval.sh $sre19_eval_root data
  # Prepare SRE21 dev
  hyp_utils/conda_env.sh \
    local/prepare_sre21av_dev_visual.py \
    --corpus-dir $sre21_dev_root \
    --output-path data/sre21_visual_dev

  hyp_utils/conda_env.sh \
    local/prepare_sre21av_eval_visual.py \
    --corpus-dir $sre21_eval_root \
    --output-path data/sre21_visual_eval

fi

if [ $stage -le 2 ];then
    local/make_janus_core.sh $janus_root data
fi

