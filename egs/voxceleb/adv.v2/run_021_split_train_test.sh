#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

attack_dir=exp/attacks/$spknet_name/
mkdir -p $attack_dir/pool_v1

if [ $stage -le 1 ];then
    echo "concatenate infos of all attacks types"
    for attack in fgsm iter-fgsm pgd-linf pgd-l1 pgd-l2 cw-l2 cw-linf cw-l0
    do
	for name in voxceleb2cat_train
	do
	    cat $attack_dir/$attack/$name/info/info.yaml
	done | awk '/attack_type:/ { sub(/pgd/,"'$attack'",$0) }
                               { print $0 }' | awk '{ print $0} 
                                                   /key_benign/ { sub(/-'$attack'-benign$/,"",$2); print "  key_original:",$2 }'
    done | awk '!/\{\}/' > $attack_dir/pool_v1/info.yaml
fi

conda activate $HYP_ENV

if [ $stage -le 2 ];then
    echo "Explit attacks and benign signals into train/val/test"
    # split attacks into train/val/test
    # signals used to train xvector extractor will be splitted 
    # into 90% train / 10% val
    # signals used to validate x-vector extractor will be used for test
    local/split_train_test.py \
	--attack-info-file $attack_dir/pool_v1/info.yaml \
	--train-list  data/voxceleb2cat_train_proc_audio_no_sil/lists_xvec/train.scp \
	--test-list  data/voxceleb2cat_train_proc_audio_no_sil/lists_xvec/val.scp \
	--p-val 0.1 \
	--output-dir $attack_dir/pool_v1

fi

if [ $stage -le 3 ];then
    echo "Make lists for attack type classification"
    local/make_train_test_lists_exp_attack_type_v1.py \
	--input-dir $attack_dir/pool_v1 \
	--output-dir data/$attack_type_split_tag $attack_type_split_opts
fi

if [ $stage -le 4 ];then
    echo "Make lists for attack SNR classification"
    local/make_train_test_lists_exp_attack_snr_v1.py \
	--input-dir $attack_dir/pool_v1 \
	--output-dir data/$snr_split_tag $snr_split_opts
fi

if [ $stage -le 5 ];then
    echo "Make lists for threat model classification"
    local/make_train_test_lists_exp_attack_threat_model_v1.py \
	--input-dir $attack_dir/pool_v1 \
	--output-dir data/$threat_model_split_tag $threat_model_split_opts
fi
