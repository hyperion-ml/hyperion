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
max_systems=8
stage=1
. parse_options.sh || exit 1;

be21=pca0.5_rmu100000_rs100000_splday150_adapt_wmu0.5_wb0.75_ww0.5_v2
be31=pca0.5_rmu100000_rs100000_splday150_adapt1_wmu0.75_wb0.5_ww0.25_adapt2_wmu0.25_wb0.25_ww0.25_v3
be31v=pca0.85_rmu100_rs100000_splday150_adapt1_wmu0.5_wb0.25_ww0.25_adapt2_wmu0.75_wb0.25_ww0.25_v3

be2resnet34open=pca0.7_rmu100000_rs100000_splday150_adapt_wmu0.5_wb0.75_ww0.5_v2
be2res2netopen=pca0.75_rmu100000_rs100000_splday150_adapt_wmu0.5_wb0.75_ww0.5_v2
be2tseres2netopen=pca0.85_rmu100000_rs100000_splday150_adapt_wmu0.5_wb0.75_ww0.5_v2
be2tseres2netallopen=pca0.7_rmu100000_rs100000_splday150_adapt_wmu0.5_wb0.75_ww0.5_v2

system_names="res2nets8-16k-chatt-b21 res2nets8-16k-chatt-b21s tseres2nets4-16k-chatt-b21 tseres2nets4-16k-chatt-b21s ecapa-16k-chatt-b21 ecapa-16k-chatt-b21s \
res2net8s-8k-chatt-b21 res2net8s-8k-chatt-b21s ecapa-8k-chatt-b21 ecapa-8k-chatt-b21s \
tseres2nets4-16k-chatt-b31 res2nets4-16k-vox-b31v tseres2nets4-16k-vox-b31v mitll1 \
tseres2net50-16k-irl-be21 tseres2net50-16k-irl-be21s res2net50-16k-bwe-v5-be21 res2net50-16k-bwe-v5-be21s \
resnet34-open-be2 res2net-open-be2 tseres2net-open-be2 tseres2net-open-all-be2 \
resnet34-open-be2s res2net-open-be2s tseres2net-open-be2s tseres2net-open-all-be2s"
system_dirs="exp/scores/fbank80_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
exp/scores/fbank80_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_snorm_v1_5000_cal_v1 \
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_snorm_v1_5000_cal_v1 \
exp/scores/fbank80_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
exp/scores/fbank80_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_snorm_v1_5000_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_snorm_v1_5000_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_ecapatdnn2048x4_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be21/plda_snorm_v1_5000_cal_v1 \
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31/plda_cal_v1 \
exp/scores/fbank80_stmn_res2net50w26s4_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1.voxcelebcat.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31v/plda_cal_v1 \
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1.voxcelebcat.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.v1/$be31v/plda_cal_v1 \
exp/scores_ll/LL_sys1 \
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.reg_lang.0m3.v1.sampler.self/$be21/plda_cal_v1 \
exp/scores/fbank80_stmn_tseres2net50w26s4_r256_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_15_arcm0.5_sgdcos_lr0.01_b128_amp.reg_lang.0m3.v1.sampler.self/$be21/plda_snorm_v1_5000_cal_v1 \
exp/scores/fbank80_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1_16_BWE-2v2_16_BWE-2v5/$be21/plda_cal_v1 \
exp/scores/fbank80_stmn_res2net50w26s8_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.02_b512_amp.v1.ft_10_10_arcm0.5_sgdcos_lr0.01_b128_amp.v1_16_BWE-2v2_16_BWE-2v5/$be21/plda_snorm_v1_5000_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_resnet34_eina_hln_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v3/$be2resnet34open/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_res2net50w26s4_eina_hln_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_20_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2res2netopen/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_tseres2net50w26s4_r256_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_16_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2tseres2netopen/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_tseres2net50w26s4_r256_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b510_amp.v1.alllangs_nocveng.ft_10_15_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2tseres2netallopen/plda_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_resnet34_eina_hln_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v3/$be2resnet34open/plda_snorm_v1_5000_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_res2net50w26s4_eina_hln_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_20_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2res2netopen/plda_snorm_v1_5000_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_tseres2net50w26s4_r256_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_16_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2tseres2netopen/plda_snorm_v1_5000_cal_v1 \
../v1.8k/exp/scores/fbank64_stmn_tseres2net50w26s4_r256_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b510_amp.v1.alllangs_nocveng.ft_10_15_arcm0.3_sgdcos_lr0.05_b128_amp.v2/$be2tseres2netallopen/plda_snorm_v1_5000_cal_v1"

for s in $system_dirs
do
  echo $s
  #local/score_sre21_official.sh $sre21_dev_root audio dev $s
  #head -n 5 $s/sre21_audio_dev_results.csv
  #head -n 2 $s/sre_cts_superset_dev_results.csv
  #grep -e ENG_YUE -e dataset $s/sre_cts_superset_dev_results.csv
  #paste $s/sre16_eval40_yue_results.csv $s/sre_cts_superset_dev_results.csv | grep -e all -e dataset
  #paste $s/sre16_eval40_yue_results.csv $s/sre_cts_superset_dev_results.csv | grep -e all | awk -F "," 'BEGIN{OFS=" & "} { print $3,$8,$9,$13,$18,$19}'
  awk -F " " 'BEGIN{OFS=" & "} /Audio/ { print $2,$3,$4}' $s/sre21_audio_dev_official_results
done
exit
output_dir=exp/fusion/v2.5.1_open_fus_pfus${p_fus}_l2${fus_l2_reg}_pcal${p_cal}_l2${cal_l2_reg}

if [ $stage -le 1 ];then
  local/fusion_sre21av_v2.1.sh \
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
	  local/score_sre21.sh data/sre21_audio_dev_test audio_dev $output_dir/$i 
	  local/score_sre21.sh data/sre21_audio-visual_dev_test audio-visual_dev $output_dir/$i
	  local/score_sre21_official.sh $sre21_dev_root audio dev $output_dir/$i 
	fi
    done
    exit
fi

# local/fusion_sanity.sh \
#   --cmd "$train_cmd --mem 24G" \
#   "$system_names" "$system_dirs" $output_dir

local/fusion_sanity.sh \
  --cmd "$train_cmd --mem 24G" \
  "$system_names f1 f4 f6 f8" "$system_dirs $output_dir/0 $output_dir/3 $output_dir/5 $output_dir/7" $output_dir
