#!/bin/bash
. ./cmd.sh
. ./path.sh
. ./datapath.sh
set -e

p_fus=0.1
p_cal=0.05
p_eval="0.01 0.05"
fus_l2_reg=1e-3
cal_l2_reg=1e-4
max_systems=5
stage=1
. parse_options.sh || exit 1;

be21=pca0.5_rmu100000_rs100000_splday150_adapt_wmu0.5_wb0.75_ww0.5_v2
be31=pca0.5_rmu100000_rs100000_splday150_adapt1_wmu0.75_wb0.5_ww0.25_adapt2_wmu0.25_wb0.25_ww0.25_v3
be31v=pca0.85_rmu100_rs100000_splday150_adapt1_wmu0.5_wb0.25_ww0.25_adapt2_wmu0.75_wb0.25_ww0.25_v3

# system_names="res2nets8-16k-chatt-b21 tseres2nets4-16k-chatt-b21 ecapa-16k-chatt-b21 res2net8s-8k-chatt-b21 ecapa-8k-chatt-b21 res2nets8-16k-chatt-b31 tseres2nets4-16k-chatt-b31 ecapa-16k-chatt-b31 res2nets4-16k-vox-b31v res2net8s-8k-chatt-b31 ecapa-8k-chatt-b31" 
# system_dirs="exp/scores/fbank80_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
# exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
# exp/scores/fbank80_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
# ../v1.8k/exp/scores/fbank64_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
# ../v1.8k/exp/scores/fbank64_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1
# exp/scores/fbank80_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31/plda_cal_v1 \
# exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31/plda_cal_v1 \
# exp/scores/fbank80_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31/plda_cal_v1 \
# exp/scores/fbank80_stmn_res2net50w26s4_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1.voxcelebcat.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31v/plda_cal_v1 \
# ../v1.8k/exp/scores/fbank64_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31/plda_cal_v1 \
# ../v1.8k/exp/scores/fbank64_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31/plda_cal_v1"

system_names="res2nets8-16k-chatt-b21 tseres2nets4-16k-chatt-b21 ecapa-16k-chatt-b21 res2net8s-8k-chatt-b21 ecapa-8k-chatt-b21 tseres2nets4-16k-chatt-b31 res2nets4-16k-vox-b31v"
system_dirs="exp/scores/fbank80_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
exp/scores/fbank80_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31/plda_cal_v1 \
exp/scores/fbank80_stmn_res2net50w26s4_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1.voxcelebcat.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31v/plda_cal_v1"


# for s in $system_dirs
# do
#   echo $s
#   #head -n 5 $s/sre21_audio_dev_results.csv
#   #head -n 2 $s/sre_cts_superset_dev_results.csv
#   #grep -e ENG_YUE -e dataset $s/sre_cts_superset_dev_results.csv
# done
# exit
output_dir=exp/fusion/v2.2_fus_pfus${p_fus}_l2${fus_l2_reg}_pcal${p_cal}_l2${cal_l2_reg}

if [ $stage -le 1 ];then
  local/fusion_sre21av_v2.sh \
    --cmd "$train_cmd --mem 24G" \
    --l2-reg $fus_l2_reg --p-fus $p_fus \
    --max-systems $max_systems --p-eval "$p_eval" \
    --p-cal $p_cal --l2-reg-cal $cal_l2_reg \
    "$system_names" "$system_dirs" $output_dir
fi

if [ $stage -le 2 ];then
    for((i=0;i<$max_systems;i++))
    do
      if [ -d $output_dir/$i ];then
	  local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $output_dir/$i 
	  local/score_sre_cts_superset.sh data/sre_cts_superset_16k_dev $output_dir/$i 
	  local/score_sre21.sh data/sre21_audio_dev_test audio_dev $output_dir/$i 
	  local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev $output_dir/$i 
	fi
    done
fi

