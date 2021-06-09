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
    echo "Split list into known and unknown attacks"
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
    echo "Make train list for known attacks signatures by attack type"
    local/make_train_test_lists_exp_attack_type_v1.py \
	--input-dir $pool_known_dir \
	--output-dir data/$sk_attack_type_split_tag $attack_type_split_opts
fi

if [ $stage -le 3 ];then
    echo "Make train list for known attacks signatures by SNR"
    local/make_train_test_lists_exp_attack_snr_v1.py \
	--input-dir $pool_known_dir \
	--output-dir data/$sk_snr_split_tag $snr_split_opts
fi

if [ $stage -le 4 ];then
    echo "Make train list for known attacks signatures by threat model"
    local/make_train_test_lists_exp_attack_threat_model_v1.py \
	--input-dir $pool_known_dir \
	--output-dir data/$sk_threat_model_split_tag $threat_model_split_opts
fi

if [ $stage -le 5 ];then
    echo "Make attack verification trials by attack type"
    for nes in 1 3 5 10 30 #50 100
    do
	output_dir=data/${attack_type_verif_split_tag}_enr${nes}sides
	$train_cmd --mem 4G $output_dir/make.log \
	hyp_utils/conda_env.sh \
	local/make_trials_exp_attack_type_verif_v2.py \
	    --input-dir $pool_all_dir \
	    --known-attacks benign $known_attacks \
	    --output-dir $output_dir \
	    --num-enroll-sides $nes $verif_split_opts & 
    done
    wait
fi

if [ $stage -le 6 ];then
    echo "Make attack verification trials by SNR"
    for nes in 1 3 5 10 30 #50 100
    do
	output_dir=data/${snr_verif_split_tag}_enr${nes}sides
	$train_cmd --mem 4G $output_dir/make.log \
	hyp_utils/conda_env.sh \
	local/make_trials_exp_attack_snr_verif_v2.py \
	    --input-dir $pool_all_dir \
	    --known-attacks benign $known_attacks \
	    --output-dir $output_dir \
	    --num-enroll-sides $nes $verif_split_opts &
    done
    wait
fi

if [ $stage -le 7 ];then
    echo "Make attack verification trials by threat_model"
    for nes in 1 3 5 10 30 #50 100
    do
	output_dir=data/${threat_model_verif_split_tag}_enr${nes}sides
	$train_cmd --mem 4G $output_dir/make.log \
	hyp_utils/conda_env.sh \
	local/make_trials_exp_attack_threat_model_verif_v2.py \
	    --input-dir $pool_all_dir \
	    --known-attacks benign $known_attacks \
	    --output-dir $output_dir \
	    --num-enroll-sides $nes $verif_split_opts &
    done
    wait
fi

if [ $stage -le 8 ];then
    echo "Make trials for novelty detection"
    $train_cmd --mem 4G data/$novelty_split_tag/make.log \
	hyp_utils/conda_env.sh \
	local/make_trials_exp_attack_type_novelty_v2.py \
	--input-dir $attack_dir/pool_v1 \
	--known-attacks benign $known_attacks \
	--output-dir data/$novelty_split_tag $novelty_split_opts 
fi
