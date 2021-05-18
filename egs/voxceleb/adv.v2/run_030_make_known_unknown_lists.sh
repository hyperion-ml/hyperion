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

attack_dir=exp/attacks/$spknet_name
pool_all_dir=$attack_dir/pool_v1
pool_known_dir=$attack_dir/pool_v1_known
mkdir -p $pool_known_dir

conda activate $HYP_ENV

if [ $stage -le 1 ];then
    for file in info.yaml test_attack_info.yaml train_attack_info.yaml val_attack_info.yaml
    do
	$train_cmd $pool_known_dir/$file.log \
	    hyp_utils/conda_env.sh local/filter_attacks.py \
	    --field attack_tag --keep $known_attacks \
	    --input-file $pool_all_dir/$file \
	    --output-file $pool_known_dir/$file &
    done
    wait
fi

# make training list for signatures with known attacks
if [ $stage -le 2 ];then
    local/make_train_test_lists_exp_attack_type_v1.py \
	--input-dir $pool_known_dir \
	--output-dir data/$k_attack_type_split_tag $attack_type_split_opts
fi

if [ $stage -le 3 ];then
    local/make_train_test_lists_exp_attack_snr_v1.py \
	--input-dir $pool_known_dir \
	--output-dir data/$k_snr_split_tag $attack_type_split_opts
fi

if [ $stage -le 4 ];then
    local/make_train_test_lists_exp_attack_threat_model_v1.py \
	--input-dir $pool_known_dir \
	--output-dir data/$k_threat_model_split_tag $attack_type_split_opts
fi


if [ $stage -le 5 ];then
    local/make_trials_exp_attack_type_verif_v2.py \
	--input-dir $pool_all_dir \
	--known-attacks benign $known_attacks \
	--output-dir data/$attack_type_verif_split_tag
fi
exit
if [ $stage -le 7 ];then
    local/make_trials_exp_attack_snr_verif_v2.py \
	--input-dir $attack_dir/pool_v1 \
	--seen-attacks benign $seen_attacks \
	--benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	--output-dir data/exp_attack_snr_verif_v2
    exit
fi

if [ $stage -le 8 ];then
    local/make_trials_exp_attack_threat_model_verif_v2.py \
	--input-dir $attack_dir/pool_v1 \
	--seen-attacks benign $seen_attacks \
	--benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	--output-dir data/exp_attack_threat_model_verif_v2
fi


if [ $stage -le 9 ];then
    for nes in 1 3 5 10 30 50 100
    do
	$train_cmd --mem 4G data/exp_attack_type_verif_${nes}s_v2/make.log \
	    local/make_trials_exp_attack_type_verif_v2.py \
	    --input-dir $attack_dir/pool_v1 \
	    --seen-attacks benign $seen_attacks \
	    --benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	    --num-enroll-sides $nes \
	    --output-dir data/exp_attack_type_verif_${nes}s_v2 &
    done
    wait
fi

if [ $stage -le 10 ];then
    for nes in 1 3 5 10 30 50 100
    do
	$train_cmd --mem 4G data/exp_attack_snr_verif_${nes}s_v2/make.log \
	    local/make_trials_exp_attack_snr_verif_v2.py \
	    --input-dir $attack_dir/pool_v1 \
	    --seen-attacks benign $seen_attacks \
	    --benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	    --num-enroll-sides $nes \
	    --output-dir data/exp_attack_snr_verif_${nes}s_v2 &
    done
    wait
fi


if [ $stage -le 11 ];then
    for nes in 1 3 5 10 30 50 100
    do
	$train_cmd --mem 4G data/exp_attack_threat_model_verif_${nes}s_v2/make.log \
	    local/make_trials_exp_attack_threat_model_verif_v2.py \
	    --input-dir $attack_dir/pool_v1 \
	    --seen-attacks benign $seen_attacks \
	    --benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	    --num-enroll-sides $nes \
	    --output-dir data/exp_attack_threat_model_verif_${nes}s_v2 &
    done
    wait
fi


if [ $stage -le 12 ];then
    $train_cmd --mem 4G data/exp_attack_type_verif_novely_v2/make.log \
	local/make_trials_exp_attack_type_novelty_v2.py \
	--input-dir $attack_dir/pool_v1 \
	--seen-attacks benign $seen_attacks \
	--benign-wav-file data/voxceleb2cat_proc_audio_no_sil/wav.scp \
	--output-dir data/exp_attack_type_novelty_v2 
fi
