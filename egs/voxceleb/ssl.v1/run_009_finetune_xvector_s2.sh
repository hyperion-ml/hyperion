#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=4
config_file=default_config.sh
interactive=false
. parse_options.sh || exit 1;

./run_007_train_xvector.sh \
  --config-file $config_file \
  --ngpu $ngpu \
  --stage $stage \
  --ft-stage 2 \
  --interactive $interactive

