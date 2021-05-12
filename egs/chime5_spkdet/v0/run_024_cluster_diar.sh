#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

stage=1

lda_dim=120
net_name=1a
plda_data=voxceleb

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors_diar/$net_name

be_name=lda${lda_dim}_plda_${plda_data}
be_dir=exp/be_diar/$net_name/$be_name
score_dir=exp/diarization/$net_name/$be_name


# Cluster the PLDA scores using a stopping threshold.
if [ $stage -le 1 ]; then
    # First, we loop the threshold to get different diarizations
    # Best threshold will be the one that produces minimum cprimary in speaker verification task.
    mkdir -p $score_dir/tuning
    for dataset in chime5_spkdet_test
    do
	echo "Tuning clustering threshold for $dataset"
	for threshold in -2.0 -1.5 -1.2 -1.1 -1 -0.9 -0.8 -0.7 -0.5
	do
	    steps_kaldi_diar/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
					--threshold $threshold $score_dir/$dataset/plda_scores \
					$score_dir/$dataset/plda_scores_t$threshold

	done
    done
  exit
fi
