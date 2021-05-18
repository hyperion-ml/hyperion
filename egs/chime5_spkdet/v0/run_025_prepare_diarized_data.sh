#!/bin/bash
# Copyright     2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

diar_thr=-0.9
min_dur=10
rttm_dir=./exp/diarization/1a/lda120_plda_voxceleb
diar_name=diar1a${diar_thr}

stage=1

. parse_options.sh || exit 1;

vaddiardir=`pwd`/vad_${diar_name}

if [ $stage -le 1 ]; then
    # Create datasets with diarization
    for name in chime5_spkdet_test
    do
	rttm=$rttm_dir/$name/plda_scores_t${diar_thr}/rttm
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur data/$name $rttm data/${name}_${diar_name} $vaddiardir
    done

fi

if [ $stage -le 2 ]; then
    for name in chime5_spkdet_test
    do
	local/make_diar_trials.sh data/${name}_${diar_name}/orig2utt data/$name/trials data/${name}_${diar_name}/trials
    done
fi

exit
