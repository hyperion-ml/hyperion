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

net_name=1a
num_epochs=2
nnet_dir=exp/xvector_nnet_$net_name

stage=1

. parse_options.sh || exit 1;

if [ $stage -le 8 ]; then
    steps_kaldi_xvec/run_xvector_1a.sh --stage $stage --train-stage -1 --num_epochs $num_epochs \
				       --storage_name sre18-v1.8k-$(date +'%m_%d_%H_%M') \
				       --data data/train_combined_no_sil --nnet-dir $nnet_dir \
				       --egs-dir $nnet_dir/egs
fi

exit
