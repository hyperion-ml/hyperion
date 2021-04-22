#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e
stage=1

. parse_options.sh || exit 1;

if [ $stage -le 1 ];then
    steps_proj/eval-classif-perf-unseen-attacks.py \
	--score-file scp:exp/logits/fbank80_stmn_lresnet34_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1/exp_attack_type_v1_ASR_Espresso/test/logits.scp \
	--key-file data/exp_attack_type_v1_ASR_Espresso/all_utt2attack \
	--class-file data/exp_attack_type_v1/class2int 
fi

if [ $stage -le 2 ];then

    steps_proj/eval-classif-perf-plda-unseen-attacks.py \
	--v-file scp:exp/signatures/fbank80_stmn_lresnet34_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1/exp_attack_type_v1_ASR_Espresso/test/xvector.scp \
	--key-file data/exp_attack_type_v1_ASR_Espresso/all_utt2attack \
	--class-file data/exp_attack_type_v1/class2int 
fi

if [ $stage -le 3 ];then

    steps_proj/eval-classif-perf-plda-unseen-attacks-noimp.py \
	--v-file scp:exp/signatures/fbank80_stmn_lresnet34_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1/exp_attack_type_v1_ASR_Espresso/test/xvector.scp \
	--key-file data/exp_attack_type_v1_ASR_Espresso/all_utt2attack \
	--class-file data/exp_attack_type_v1/class2int 
fi

