#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nj=10
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $stage -le 1 ];then
  $train_cmd \
    $token_dir/train_sp.log \
    hyperion-train-tokenizer sentencepiece \
    --cfg $token_cfg \
    --segments-file data/$token_train_data/segments.csv \
    --tokenizer-path $token_dir
     
fi
