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
use_gpu=false
xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

xvector_dir=exp/xvectors/$spknet_name
score_dir=exp/scores/$spknet_name
cal_file=$score_dir/cosine_cal_v1/cal_tel.h5
nnet=$spknet
attack_dir=exp/attacks/$spknet_name

#thresholds for p=(0.05,0.01,0.001) -> thr=(2.94, 4.60, 6.90)
thr005=2.94
thr001=4.60
thr0001=6.90
threshold=$thr005

if [ $stage -le 1 ]; then
    # create fgsm attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 25 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type fgsm --attacks.min-eps 3e-6 --attacks.max-eps 0.03 --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --random-seed 1000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	fgsm \
    	$attack_dir/fgsm/voxceleb1_test

fi

if [ $stage -le 3 ]; then
    # create iter fgsm attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type iter-fgsm --attacks.min-eps 3e-6 --attacks.max-eps 0.03 --attacks.min-alpha 1e-6 --attacks.max-alpha 0.005 --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --random-seed 3000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	iter-fgsm \
    	$attack_dir/iter-fgsm/voxceleb1_test
fi

if [ $stage -le 4 ]; then
    # create pgd attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type pgd --attacks.norms inf --attacks.min-eps 3e-6 --attacks.max-eps 0.03 --attacks.min-alpha 1e-6 --attacks.max-alpha 0.005 --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --attacks.min-num-random-init 2 --attacks.max-num-random-init 5 --attacks.min-iter 10 --attacks.max-iter 100 --random-seed 4000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	pgd-linf \
    	$attack_dir/pgd-linf/voxceleb1_test
fi

if [ $stage -le 5 ]; then
    # create pgd attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type pgd --attacks.norms 1 --attacks.min-eps 3e-6 --attacks.max-eps 0.03 --attacks.min-alpha 1e-6 --attacks.max-alpha 0.005 --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --attacks.min-num-random-init 2 --attacks.max-num-random-init 5  --attacks.min-iter 10 --attacks.max-iter 100 --attacks.norm-time --random-seed 5000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	pgd-l1 \
    	$attack_dir/pgd-l1/voxceleb1_test
fi

if [ $stage -le 6 ]; then
    # create pgd attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 250 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type pgd --attacks.norms 2 --attacks.min-eps 3e-6 --attacks.max-eps 0.03 --attacks.min-alpha 1e-6 --attacks.max-alpha 0.005 --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --attacks.min-num-random-init 2 --attacks.max-num-random-init 5  --attacks.min-iter 10 --attacks.max-iter 100 --attacks.norm-time  --random-seed 6000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	pgd-l2 \
    	$attack_dir/pgd-l2/voxceleb1_test
fi


if [ $stage -le 7 ]; then
    # create CW-L2 attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 500 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type cw-l2 --attacks.min-confidence 0 --attacks.max-confidence 3 --attacks.min-lr 1e-5 --attacks.max-lr 1e-3 --attacks.min-iter 10 --attacks.max-iter 200 --attacks.norm-time --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --random-seed 7000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	cw-l2 \
    	$attack_dir/cw-l2/voxceleb1_test

fi

if [ $stage -le 8 ]; then
    # create CW-Linf attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 500 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type cw-linf --attacks.min-confidence 0 --attacks.max-confidence 3 --attacks.min-lr 1e-5 --attacks.max-lr 1e-3 --attacks.min-iter 10 --attacks.max-iter 200 --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --random-seed 8000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	cw-linf \
    	$attack_dir/cw-linf/voxceleb1_test
fi

if [ $stage -le 9 ]; then
    # create CW-L0 attacks
    hyp_utils/adv/generate_adv_attacks_xvector_verif.sh  --cmd "$xvec_cmd --mem 12G" --nj 2000 ${xvec_args} \
	--use-bin-vad false \
	--feat-config $feat_config \
	--attacks-opts "--attacks.attack-type cw-l0 --attacks.min-confidence 0 --attacks.max-confidence 3 --attacks.min-lr 1e-5 --attacks.max-lr 1e-3 --attacks.min-iter 10 --attacks.max-iter 100 --p-tar-attack $p_tar_attack --p-non-attack $p_non_attack --random-seed 9000 $spkv_attacks_common_opts" \
	--cal-file $cal_file --threshold $threshold \
    	$nnet \
	data/voxceleb1_test/trials_o_clean \
    	data/voxceleb1_test/utt2model \
        data/voxceleb1_test \
    	$xvector_dir/voxceleb1_test/xvector.scp \
	cw-l0 \
    	$attack_dir/cw-l0/voxceleb1_test
fi

exit
