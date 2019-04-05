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

stage=1
diar_name=track1a

. parse_options.sh || exit 1;

nnet_dir=exp/xvector_nnet_$net_name
xvector_dir=exp/xvectors/$net_name


if [ $stage -le 1 ]; then
    # Extract xvectors for tracking data

    for name in chime5_spkdet_test_${diar_name} 
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 30 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done


fi
    
exit
