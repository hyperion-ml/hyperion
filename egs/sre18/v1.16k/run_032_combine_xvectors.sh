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

config_file=default_config.sh
stage=1

. parse_options.sh || exit 1;
. $config_file

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ]; then
    
    utils/combine_data.sh data/sitw_dev data/sitw_dev_enroll data/sitw_dev_test
    mkdir -p $xvector_dir/sitw_dev
    cat $xvector_dir/sitw_dev_{enroll,test}/xvector.scp > $xvector_dir/sitw_dev/xvector.scp

    mkdir -p $xvector_dir/sitw_eval
    cat $xvector_dir/sitw_eval_{enroll,test}/xvector.scp > $xvector_dir/sitw_eval/xvector.scp

    mkdir -p $xvector_dir/sitw_dev_${diar_name}
    cat $xvector_dir/sitw_dev_{enroll,test_${diar_name}}/xvector.scp > $xvector_dir/sitw_dev_${diar_name}/xvector.scp

    mkdir -p $xvector_dir/sitw_eval_${diar_name}
    cat $xvector_dir/sitw_eval_{enroll,test_${diar_name}}/xvector.scp > $xvector_dir/sitw_eval_${diar_name}/xvector.scp
    
    utils/combine_data.sh data/sitw_dev1s_${diar_name} data/sitw_train_dev data/sitw_dev_test_${diar_name}
    mkdir -p $xvector_dir/sitw_dev1s_${diar_name}
    cat $xvector_dir/sitw_{train_dev,dev_test_${diar_name}}/xvector.scp > $xvector_dir/sitw_dev1s_${diar_name}/xvector.scp

    utils/combine_data.sh data/sre18_dev_vast data/sre18_dev_enroll_vast data/sre18_dev_test_vast 
    mkdir -p $xvector_dir/sre18_dev_vast
    cat $xvector_dir/sre18_dev_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_dev_vast/xvector.scp

fi

if [ $stage -le 2 ]; then
    utils/combine_data.sh data/sre18_dev_vast_${diar_name} data/sre18_dev_enroll_vast data/sre18_dev_test_vast_${diar_name} 
    mkdir -p $xvector_dir/sre18_dev_vast_${diar_name}
    cat $xvector_dir/sre18_dev_{enroll_vast,test_vast_${diar_name}}/xvector.scp > $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp
    
    utils/combine_data.sh data/sitw_sre18_dev_vast_${diar_name} data/sitw_dev1s_${diar_name} data/sre18_dev_vast_${diar_name} 
    mkdir -p $xvector_dir/sitw_sre18_dev_vast_${diar_name}
    cat $xvector_dir/sitw_dev1s_${diar_name}/xvector.scp $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp > $xvector_dir/sitw_sre18_dev_vast_${diar_name}/xvector.scp

    mkdir -p $xvector_dir/sre18_eval_vast_${diar_name}
    cat $xvector_dir/sre18_eval_{enroll_vast,test_vast_${diar_name}}/xvector.scp > $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp

fi

if [ $stage -le 3 ]; then

    mkdir -p $xvector_dir/sre18_dev_cmn2
    cat $xvector_dir/sre18_dev_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_dev_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre18_dev_vast
    cat $xvector_dir/sre18_dev_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_dev_vast/xvector.scp

    mkdir -p $xvector_dir/sre18_eval_cmn2
    cat $xvector_dir/sre18_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_eval_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre18_eval_vast
    cat $xvector_dir/sre18_eval_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_eval_vast/xvector.scp

fi

exit
