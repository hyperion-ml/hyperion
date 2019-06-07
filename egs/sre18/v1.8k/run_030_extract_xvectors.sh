#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in sre_tel sre_phnmic voxceleb
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 60 \
					     $nnet_dir data/${name}_combined \
					     $xvector_dir/${name}_combined
    done

fi

if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in sre16_eval_enroll sre16_eval_test sre16_major sre16_minor \
   				  sitw_train_dev sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test \
  				  sre18_dev_unlabeled sre18_dev_enroll_cmn2 sre18_dev_test_cmn2 \
				  sre18_eval_enroll_cmn2 sre18_eval_test_cmn2 \
				  sre18_eval_enroll_vast sre18_eval_test_vast

    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done

    for name in sre16_dev_enroll sre16_dev_test sre18_dev_enroll_vast sre18_dev_test_vast
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 1 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done

fi

if [ $stage -le 3 ]; then
    mkdir -p $xvector_dir/train_combined
    cat $xvector_dir/{sre_tel,sre_phnmic,voxceleb}_combined/xvector.scp > $xvector_dir/train_combined/xvector.scp
fi

exit
