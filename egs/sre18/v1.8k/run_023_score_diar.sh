#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

xvector_dir=exp/xvectors_diar/$nnet_name

be_dir=exp/be_diar/$nnet_name/$be_diar_name
score_dir=exp/diarization/$nnet_name/$be_diar_name


# Perform PLDA scoring
if [ $stage -le 1 ]; then

    cp $xvector_dir/${plda_diar_data}_128k/mean.vec $be_dir
    # Perform PLDA scoring on all pairs of segments for each recording.
    for name in sitw_dev_test sitw_eval_test sre18_eval_test_vast
    do
	mkdir -p $score_dir/$name
	steps_kaldi_diar/score_plda.sh --cmd "$train_cmd --mem 4G" \
						--nj 20 $be_dir $xvector_dir/$name \
						$score_dir/$name/plda_scores
    done

    for name in sre18_dev_test_vast
    do
	mkdir -p $score_dir/$name
	steps_kaldi_diar/score_plda.sh --cmd "$train_cmd --mem 4G" \
						--nj 1 $be_dir $xvector_dir/$name \
						$score_dir/$name/plda_scores
    done
fi
