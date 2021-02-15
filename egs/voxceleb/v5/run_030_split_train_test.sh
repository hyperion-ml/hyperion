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

attack_dir=exp/attacks/$nnet_name/
mkdir -p $attack_dir/pool_v1

if [ $stage -le 1 ];then
    # concatenate infos of all attacks types
    for attack in fgsm rand-fgsm iter-fgsm cw-l2 cw-linf cw-l0 pgd-linf pgd-l1 pgd-l2
    do
	for name in voxceleb2cat
	do
	    cat $attack_dir/$attack/$name/info/info.yml
	done | awk '/attack_type:/ { sub(/pgd/,"'$attack'",$0); sub(/rand-fgsm/,"fgsm",$0) }
                               { print $0 }'
    done | awk '!/\{\}/' > $attack_dir/pool_v1/info.yml
fi


if [ $stage -le 2 ];then
    # split attacks into train/val/test
    # signals used to train xvector extractor will be splitted 
    # into 90% train / 10% val
    # signals used to validate x-vector extractor will be used for test
    local/split_train_test.py \
	--attack-info-file $attack_dir/pool_v1/info.yml \
	--train-list  data/voxceleb2cat_proc_audio_no_sil/lists_xvec/train.scp \
	--test-list  data/voxceleb2cat_proc_audio_no_sil/lists_xvec/val.scp \
	--p-val 0.1 \
	--output-dir $attack_dir/pool_v1
fi

if [ $stage -le 3 ];then
    local/make_train_test_lists_exp_attack_type_v1.py \
	--input-dir $attack_dir/pool_v1 \
	--benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	--benign-durs data/voxceleb2cat_proc_audio_no_sil/utt2dur \
	--output-dir data/exp_attack_type_v1
    
fi

if [ $stage -le 4 ];then
    local/make_train_test_lists_exp_attack_snr_v1.py \
	--input-dir $attack_dir/pool_v1 \
	--benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	--benign-durs data/voxceleb2cat_proc_audio_no_sil/utt2dur \
	--output-dir data/exp_attack_snr_v1
fi

