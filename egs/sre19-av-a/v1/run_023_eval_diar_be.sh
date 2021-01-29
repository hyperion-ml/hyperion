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

datasets="sitw_dev_test sitw_eval_test \
          sre18_dev_test_vast sre18_eval_test_vast \
	  sre19_av_a_dev_test sre19_av_a_eval_test \
	  janus_dev_test_core janus_eval_test_core"

# Perform PLDA scoring
if [ $stage -le 1 ]; then

    cp $xvector_dir/${plda_diar_data}_128k/mean.vec $be_dir
    # Perform PLDA scoring on all pairs of segments for each recording.
    for name in $datasets
    do
	(
	    mkdir -p $score_dir/$name
	    num_spk=$(wc -l $xvector_dir/${name}/spk2utt | cut -d " " -f 1)
	    nj=$(($num_spk < 20 ? $num_spk:20))
	    steps_kaldi_diar/score_plda.sh --cmd "$train_cmd --mem 4G" \
		--nj $nj $be_dir $xvector_dir/$name \
		$score_dir/$name/plda_scores
	) &
    done

fi
wait

if [ $stage -le 2 ]; then
    # First, we loop the threshold to get different diarizations
    # Best threshold will be the one that produces minimum cprimary in speaker verification task.
    mkdir -p $score_dir/tuning
    for dataset in $datasets
    do
	echo "Tuning clustering threshold for $dataset"
	for threshold in -0.9 #-1.5 -1.25 -0.9 -0.5 0 0.5
	do
	    (
		num_spk=$(wc -l $score_dir/$dataset/plda_scores/spk2utt  | cut -d " " -f 1)
		nj=$(($num_spk < 20 ? $num_spk:20))
		steps_kaldi_diar/cluster.sh --cmd "$train_cmd --mem 4G" --nj $nj \
		    --threshold $threshold $score_dir/$dataset/plda_scores \
		    $score_dir/$dataset/plda_scores_t$threshold
	    ) &
	done
    done
fi
