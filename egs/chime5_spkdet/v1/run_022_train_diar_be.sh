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


#Train LDA
if [ $stage -le 1 ];then

    mkdir -p $be_dir
    # Train a LDA model on Voxceleb,
    echo "Train LDA"
    $train_cmd $be_dir/log/lda.log \
	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
	       "ark:ivector-subtract-global-mean scp:$xvector_dir/voxceleb_128k/xvector.scp ark:- |" \
	       ark:$xvector_dir/voxceleb_128k/utt2spk $be_dir/transform.mat || exit 1;

fi

# Train PLDA models
if [ $stage -le 2 ]; then
    # Train a PLDA model on Voxceleb,
    echo "Train PLDA"
    $train_cmd $be_dir/log/plda.log \
	       ivector-compute-plda ark:$xvector_dir/voxceleb_128k/spk2utt \
	       "ark:ivector-subtract-global-mean \
      scp:$xvector_dir/voxceleb_128k/xvector.scp ark:- \
      | transform-vec $be_dir/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
	       $be_dir/plda || exit 1;
    
    cp $xvector_dir/voxceleb_128k/mean.vec $be_dir

fi

