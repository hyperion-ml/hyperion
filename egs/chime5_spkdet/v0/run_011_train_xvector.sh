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

net_name=3b
num_epochs=3
nnet_dir=exp/xvector_nnet_$net_name

stage=6

. parse_options.sh || exit 1;

if [ $stage -le 8 ]; then
    steps_kaldi_xvec/run_xvector_3b.sh --stage $stage --train-stage -1 --num_epochs $num_epochs --nodes b1 \
    				       --storage_name voices_challenge-v1-$(date +'%m_%d_%H_%M') \
    				       --data data/train_combined_no_sil --nnet-dir $nnet_dir \
    				       --egs-dir $nnet_dir/egs
    # steps_kaldi_xvec/run_xvector_3b.sh --stage $stage --train-stage 2 --num_epochs $num_epochs --nodes b1 \
    # 				       --storage_name voices_challenge-v1-$(date +'%m_%d_%H_%M') \
    # 				       --data data/train_combined_no_sil --nnet-dir $nnet_dir \
    # 				       --egs-dir $nnet_dir/egs

fi

exit
