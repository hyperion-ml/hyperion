#!/bin/bash
# Copyright 2019   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

net_name=3b

stage=1

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors/$net_name

if [ $stage -le 1 ]; then

    utils/combine_data.sh data/voices19_challenge_dev data/voices19_challenge_dev_enroll data/voices19_challenge_dev_test
    mkdir -p $xvector_dir/voices19_challenge_dev
    cat $xvector_dir/voices19_challenge_dev_{enroll,test}/xvector.scp > $xvector_dir/voices19_challenge_dev/xvector.scp

    utils/combine_data.sh data/voices19_challenge_eval data/voices19_challenge_eval_enroll data/voices19_challenge_eval_test
    mkdir -p $xvector_dir/voices19_challenge_eval
    cat $xvector_dir/voices19_challenge_eval_{enroll,test}/xvector.scp > $xvector_dir/voices19_challenge_eval/xvector.scp

fi

exit
