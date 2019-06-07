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

score_dir=exp/diarization/$nnet_name/$be_diar_name

# Cluster the PLDA scores using a stopping threshold.
if [ $stage -le 1 ]; then
    # First, we loop the threshold to get different diarizations
    # Best threshold will be the one that produces minimum cprimary in speaker verification task.
    mkdir -p $score_dir/tuning
    for dataset in sitw_dev_test sitw_eval_test sre18_dev_test_vast sre18_eval_test_vast;
    do
	echo "Tuning clustering threshold for $dataset"
	for threshold in -0.9
	do
	    steps_kaldi_diar/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
					--threshold $threshold $score_dir/$dataset/plda_scores \
					$score_dir/$dataset/plda_scores_t$threshold
	done
    done
fi
