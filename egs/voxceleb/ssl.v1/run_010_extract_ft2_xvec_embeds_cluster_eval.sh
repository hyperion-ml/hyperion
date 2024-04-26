#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nnet_stage=2
config_file=default_config.sh
use_gpu=true
xvec_chunk_length=120.0
do_clustering=true
. parse_options.sh || exit 1;

./run_006_extract_dino_embeds_cluster_eval.sh \
  --config-file $config_file \
  --stage $stage \
  --nnet-stage $nnet_stage \
  --ft-stage 2 \
  --use-gpu $use_gpu \
  --xvec-chunk-length $xvec_chunk_length \
  --do-clustering $do_clustering
