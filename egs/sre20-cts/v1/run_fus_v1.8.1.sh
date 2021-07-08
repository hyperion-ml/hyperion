#!/bin/bash
. ./cmd.sh
. ./path.sh
. ./datapath.sh
set -e

fus_set=sre16-9
cal_set=sre16-yue
p_trn=0.05
p_eval=0.05
fus_l2_reg=1e-3
cal_l2_reg=1e-4
max_systems=5
stage=1
. parse_options.sh || exit 1;

be1=lda150_splday125_sre_tel_v4_adapt_coral_mu1T1_mu1B0.75W0.75_realtel_noeng/plda_snorm_realtel_alllangs500_cal_v2sre16-yue
be2=pca0.975_knn800-300_lda150_splday125_realtel_alllangs_v3_adapt_mu1B0.5W0.5/plda_cal_v2sre16-yue

# system_names="resnet34eng_ll1 resnet34allnocvcn-knn resnet34einaallnocvcn-knn tseresnet34allnocvcn-knn tb8allnocvcn-knn"
# system_dirs="exp/scores/LL_scores_for_fusion_1_cal_v2sre16-yue \
# exp/scores/fbank64_stmn_resnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2 \
# exp/scores/fbank64_stmn_resnet34_eina_hln_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2 \
# exp/scores/fbank64_stmn_tseresnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2 \
# exp/scores/fbank64_stmn_transformer_csub_lac6b8d608h8linff2432_e256_arcs30m0.3_do0_adam_lr0.005_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2"

system_names="resnet34allnocvcn-knn resnet34einaallnocvcn-knn tseresnet34allnocvcn-knn tb8allnocvcn-knn"
system_dirs="exp/scores/fbank64_stmn_resnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2 \
exp/scores/fbank64_stmn_resnet34_eina_hln_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2 \
exp/scores/fbank64_stmn_tseresnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2 \
exp/scores/fbank64_stmn_transformer_csub_lac6b8d608h8linff2432_e256_arcs30m0.3_do0_adam_lr0.005_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2"

output_dir=exp/fusion/v1.8.1_${fus_set}_ptrn${p_trn}_l2${fus_l2_reg}

if [ $stage -le 1 ];then
    local/fusion_sre20cts_v1.sh --cmd "$train_cmd --mem 24G" \
				--fus-set $fus_set \
				--l2-reg $fus_l2_reg --p-trn $p_trn \
				--max-systems $max_systems --p-eval "$p_eval" \
				"$system_names" "$system_dirs" $output_dir
fi

if [ $stage -le 2 ];then
    for((i=0;i<$max_systems;i++))
    do
	if [ -d $output_dir/$i ];then
	    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $output_dir/$i
	    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl $output_dir/$i
	    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $output_dir/$i
	fi
    done
fi

if [ $stage -le 3 ];then
    # recalibrate the fusion scores
    for((i=0;i<$max_systems;i++))
    do
	if [ -d $output_dir/$i ];then
	    local/calibrate_sre20cts_v1.sh --cmd "$train_cmd" --l2-reg $cal_l2_reg $cal_set $output_dir/${i} &
	fi
    done
    wait
fi

if [ $stage -le 4 ];then
    for((i=0;i<$max_systems;i++))
    do
	if [ -d $output_dir/$i ];then
	    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $output_dir/${i}_cal_v1${cal_set}
	    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl $output_dir/${i}_cal_v1${cal_set}
	    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $output_dir/${i}_cal_v1${cal_set}
	fi
    done
fi


