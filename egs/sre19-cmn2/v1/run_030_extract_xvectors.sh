#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
adapted=false

. parse_options.sh || exit 1;
. $config_file

if [ "$adapted" == "true" ];then
    nnet_name=${nnet_name}_adapt_cmn2
    nnet_dir=${nnet_dir}_adapt_cmn2
fi

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 2 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in sre_tel 
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 150 \
					     $nnet_dir data/${name} \
					     $xvector_dir/${name}
    done

    for name in sre18_cmn2_adapt_lab
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 50 \
					     $nnet_dir data/${name} \
					     $xvector_dir/${name}
    done

fi

if [ $stage -le 3 ]; then
    # Extracts x-vectors for evaluation
    for name in sre18_dev_unlabeled \
		    sre18_eval40_enroll_cmn2 sre18_eval40_test_cmn2 \
		    sre19_eval_enroll_cmn2 sre19_eval_test_cmn2
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 80 ? $num_spk:80))
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done
fi

if [ $stage -le 4 ]; then
    mkdir -p $xvector_dir/sre18_eval40_cmn2
    cat $xvector_dir/sre18_eval40_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_eval40_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre19_eval_cmn2
    cat $xvector_dir/sre19_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre19_eval_cmn2/xvector.scp
fi


exit
