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
be2=be_v4_thrahc0.7/cosine_cal_v2_sre21
be2s=be_v4_thrahc0.7/cosine_snorm1000_v1_cal_v2_sre21

echo "This is just a fusion example, \
     you won't be able to run it if you don't have all the systems need for the fusion"

system_names="pt-b2s carlos mxnet-b2s"
system_dirs="$nnet1/$be2s exp/scores/carlos/cosine_cal_v2_sre21 $nnet2/$be2s"

output_dir=exp/fusion/v2.4_ptrn${p_trn}_l2${fus_l2_reg}

if [ $stage -le 1 ];then
    local/fusion_sre21av_v2.1.sh --cmd "$train_cmd --mem 24G" \
				--l2-reg $fus_l2_reg --p-trn $p_trn \
				--max-systems $max_systems --p-eval "$p_eval" \
				"$system_names" "$system_dirs" $output_dir
fi

if [ $stage -le 2 ];then
  for((i=0;i<$max_systems;i++))
  do
    if [ -d $output_dir/$i ];then
      local/score_sre21av.sh data/sre21_visual_dev_test visual_dev $output_dir/$i
      local/score_sre21av.sh data/sre21_visual_eval_test visual_eval $output_dir/$i
      local/score_sre21_official.sh $sre21_dev_root visual dev $output_dir/$i
      local/score_sre21_official.sh $sre21_eval_root visual eval $output_dir/$i 
    fi
  done
fi
