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

xvector_dir=exp/xvectors/$nnet_name


if [ $stage -le 1 ]; then
    # Extract xvectors for diarization data

    for name in sitw_dev_test_${diar_name} sitw_eval_test_${diar_name} sre18_eval_test_vast_${diar_name}
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done

    for name in sre18_dev_test_vast_${diar_name}
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 1 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done

fi
    
exit
