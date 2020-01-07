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


#Train LDA
if [ $stage -le 1 ];then

    mkdir -p $be_dir
    # Train a LDA model on Voxceleb,
    echo "Train LDA"
    $train_cmd $be_dir/log/lda.log \
	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
	       "ark:ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- |" \
	       ark:$xvector_dir/${plda_diar_data}_128k/utt2spk $be_dir/transform.mat || exit 1;

fi

# Train PLDA models
if [ $stage -le 2 ]; then
    # Train a PLDA model on Voxceleb,
    echo "Train PLDA"
    $train_cmd $be_dir/log/plda.log \
	       ivector-compute-plda ark:$xvector_dir/${plda_diar_data}_128k/spk2utt \
	       "ark:ivector-subtract-global-mean \
      scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- \
      | transform-vec $be_dir/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
	       $be_dir/plda || exit 1;
    
    cp $xvector_dir/${plda_diar_data}_128k/mean.vec $be_dir

fi

