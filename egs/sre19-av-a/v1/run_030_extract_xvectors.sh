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
    for name in voxceleb #sre_tel sre_phnmic
    do
    	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 1000 \
    					     $nnet_dir data/${name}_combined \
    					     $xvector_dir/${name}_combined
    done

    for name in dihard2_train
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 200 \
					     $nnet_dir data/${name} \
					     $xvector_dir/${name}
    done


fi

if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test \
	sre18_eval_enroll_vast sre18_eval_test_vast sre18_dev_enroll_vast sre18_dev_test_vast \
	sre19_av_a_dev_enroll sre19_av_a_eval_enroll \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_enroll janus_dev_test_core janus_eval_enroll janus_eval_test_core 
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done
fi

#combine x-vectors

# if [ $stage -le 3 ]; then
#     mkdir -p $xvector_dir/train_combined
#     cat $xvector_dir/{sre_phnmic,voxceleb}_combined/xvector.scp > $xvector_dir/train_combined/xvector.scp
# fi


if [ $stage -le 3 ]; then
    
    utils/combine_data.sh data/sitw_dev data/sitw_dev_enroll data/sitw_dev_test
    mkdir -p $xvector_dir/sitw_dev
    cat $xvector_dir/sitw_dev_{enroll,test}/xvector.scp > $xvector_dir/sitw_dev/xvector.scp

    mkdir -p $xvector_dir/sitw_eval
    cat $xvector_dir/sitw_eval_{enroll,test}/xvector.scp > $xvector_dir/sitw_eval/xvector.scp

    utils/combine_data.sh data/sre18_dev_vast data/sre18_dev_enroll_vast data/sre18_dev_test_vast 
    mkdir -p $xvector_dir/sre18_dev_vast
    cat $xvector_dir/sre18_dev_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_dev_vast/xvector.scp

    utils/combine_data.sh data/sre18_eval_vast data/sre18_eval_enroll_vast data/sre18_eval_test_vast 
    mkdir -p $xvector_dir/sre18_eval_vast
    cat $xvector_dir/sre18_eval_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_eval_vast/xvector.scp

    utils/combine_data.sh data/sre19_av_a_dev data/sre19_av_a_dev_enroll data/sre19_av_a_dev_test 
    mkdir -p $xvector_dir/sre19_av_a_dev
    cat $xvector_dir/sre19_av_a_dev_{enroll,test}/xvector.scp > $xvector_dir/sre19_av_a_dev/xvector.scp

    utils/combine_data.sh data/sre19_av_a_eval data/sre19_av_a_eval_enroll data/sre19_av_a_eval_test 
    mkdir -p $xvector_dir/sre19_av_a_eval
    cat $xvector_dir/sre19_av_a_eval_{enroll,test}/xvector.scp > $xvector_dir/sre19_av_a_eval/xvector.scp

    utils/combine_data.sh data/janus_dev_core data/janus_dev_enroll data/janus_dev_test_core
    mkdir -p $xvector_dir/janus_dev_core
    cat $xvector_dir/janus_dev_{enroll,test_core}/xvector.scp > $xvector_dir/janus_dev_core/xvector.scp

    utils/combine_data.sh data/janus_eval_core data/janus_eval_enroll data/janus_eval_test_core
    mkdir -p $xvector_dir/janus_eval_core
    cat $xvector_dir/janus_eval_{enroll,test_core}/xvector.scp > $xvector_dir/janus_eval_core/xvector.scp



fi

if [ $stage -le 4 ]; then

    utils/combine_data.sh data/sitw_sre18_dev_vast data/sitw_dev data/sre18_dev_vast
    mkdir -p $xvector_dir/sitw_sre18_dev_vast
    cat $xvector_dir/{sitw_dev,sre18_dev_vast}/xvector.scp > $xvector_dir/sitw_sre18_dev_vast/xvector.scp
fi

exit
