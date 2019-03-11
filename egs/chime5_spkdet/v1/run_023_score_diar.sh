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
    for name in chime5_spkdet_test
    do
	mkdir -p $score_dir/$name
	awk '{ print $1,$2}' $xvector_dir/$name/segments > $xvector_dir/$name/utt2spk
	utils/utt2spk_to_spk2utt.pl $xvector_dir/$name/utt2spk > $xvector_dir/$name/spk2utt
	steps_kaldi_diar/score_plda.sh --cmd "$train_cmd --mem 16G" \
				       --nj 20 $be_dir $xvector_dir/$name \
				       $score_dir/$name/plda_scores
    done

fi
