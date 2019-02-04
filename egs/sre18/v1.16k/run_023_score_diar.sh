#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

stage=1

lda_dim=120
net_name=1a

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors_diar/$net_name

plda_data=voxceleb
be_name=lda${lda_dim}_plda_${plda_data}
be_dir=exp/be_diar/$net_name/$be_name
score_dir=exp/diarization/$net_name/$be_name


# Perform PLDA scoring
if [ $stage -le 1 ]; then

    cp $xvector_dir/voxceleb_128k/mean.vec $be_dir
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
    exit

fi
