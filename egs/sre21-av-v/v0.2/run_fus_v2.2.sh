#!/bin/bash
. ./cmd.sh
. ./path.sh
. ./datapath.sh
set -e

p_trn=0.05
p_eval="0.05"
fus_l2_reg=1e-4
max_systems=4
stage=1
. parse_options.sh || exit 1;

nnet1=exp/scores/r100-v4
nnet2=../v0.1/exp/scores/r100-v4
be1=be_v3/cosine_cal_v2_sre21
be1s=be_v3/cosine_snorm1000_v1_cal_v2_sre21
be2=be_v4_thrahc0.8/cosine_cal_v2_sre21
be2s=be_v4_thrahc0.8/cosine_snorm1000_v1_cal_v2_sre21
be3=be_v6_selfatt_a2/cosine_cal_v2_sre21
be3s=be_v6_selfatt_a2/cosine_snorm1000_v1_cal_v2_sre21
be4=be_v7_selfatt_a2_att_a4/cosine_cal_v2_sre21
be4s=be_v7_selfatt_a2_att_a4/cosine_snorm1000_v1_cal_v2_sre21

system_names="pt-b1s pt-b2s pt-b3s mxnet-b1s mxnet-b2s mxnet-b3s"
system_dirs="$nnet1/$be1s $nnet1/$be2s $nnet1/$be3s \
$nnet2/$be1s $nnet2/$be2s $nnet2/$be3s"

# for s in $system_dirs
# do
#   echo $s
#   #head -n 5 $s/sre21_audio_dev_results.csv
#   #head -n 2 $s/sre_cts_superset_dev_results.csv
#   #grep -e ENG_YUE -e dataset $s/sre_cts_superset_dev_results.csv
# done
# exit
output_dir=exp/fusion/v2.2_ptrn${p_trn}_l2${fus_l2_reg}

if [ $stage -le 1 ];then
    local/fusion_sre21av_v2.sh --cmd "$train_cmd --mem 24G" \
				--l2-reg $fus_l2_reg --p-trn $p_trn \
				--max-systems $max_systems --p-eval "$p_eval" \
				"$system_names" "$system_dirs" $output_dir
fi

if [ $stage -le 2 ];then
  for((i=0;i<$max_systems;i++))
  do
    if [ -d $output_dir/$i ];then
      local/score_sre21av.sh data/sre21_visual_dev_test visual_dev $output_dir/$i
      local/score_janus_core.sh data/janus_dev_test_core dev $output_dir/$i
      local/score_janus_core.sh data/janus_eval_test_core eval $output_dir/$i
    fi
  done
fi

