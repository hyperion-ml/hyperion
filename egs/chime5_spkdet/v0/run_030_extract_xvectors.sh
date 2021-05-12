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

stage=1

. parse_options.sh || exit 1;

nnet_dir=exp/xvector_nnet_$net_name
xvector_dir=exp/xvectors/$net_name

if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in voxceleb sitw_train
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 60 \
					     $nnet_dir data/${name}_combined \
					     $xvector_dir/${name}_combined
    done

fi

if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in chime5_spkdet_enroll chime5_spkdet_test chime5_spkdet_test_gtvad
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 30 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done
    exit
fi

if [ $stage -le 3 ]; then
    mkdir -p $xvector_dir/train_combined
    cat $xvector_dir/{voxceleb,sitw_train}_combined/xvector.scp > $xvector_dir/train_combined/xvector.scp

fi

exit
