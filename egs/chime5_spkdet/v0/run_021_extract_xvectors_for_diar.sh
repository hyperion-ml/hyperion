#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

stage=1
net_name=1a

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors_diar/$net_name
nnet_dir=exp/xvector_nnet_$net_name


# Extract x-vectors
if [ $stage -le 1 ]; then
    for name in chime5_spkdet_test
    do
	steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
					     --nj 30 --window 1.5 --period 0.75 --apply-cmn false \
					     --min-segment 0.5 $nnet_dir \
					     data_diar/${name}_cmn_segmented $xvector_dir/$name
    done


  # Reduce the amount of training data for the PLDA,
  utils/subset_data_dir.sh data_diar/voxceleb_cmn_segmented 128000 data_diar/voxceleb_cmn_segmented_128k
  # Extract x-vectors for the Voxceleb, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
				       --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
				       --hard-min true $nnet_dir \
				       data_diar/voxceleb_cmn_segmented_128k $xvector_dir/voxceleb_128k
  exit
fi
