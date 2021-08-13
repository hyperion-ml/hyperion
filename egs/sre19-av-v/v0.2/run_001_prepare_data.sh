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
    # Prepare sre19
    local/make_sre19av_v_dev.sh $sre19_dev_root data
    local/make_sre19av_v_eval.sh $sre19_eval_root data
fi

if [ $stage -le 2 ];then
    local/make_janus_core.sh $janus_root data
fi


