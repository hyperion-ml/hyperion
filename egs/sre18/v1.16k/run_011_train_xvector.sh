#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ $stage -le 8 ]; then
    steps_kaldi_xvec/run_xvector_${nnet_vers}.sh --stage $stage --train-stage -1 --num_epochs $nnet_num_epochs \
    				 --storage_name sre18-v1.16k-$(date +'%m_%d_%H_%M') \
    				 --data data/${nnet_data}_no_sil --nnet-dir $nnet_dir \
    				 --egs-dir $nnet_dir/egs
fi

exit
