#!/bin/bash
. ./cmd.sh
. ./path.sh
. ./datapath.sh
set -e

p_trn=0.1
p_eval="0.01 0.05"
fus_l2_reg=1e-4
max_systems=4
stage=1
. parse_options.sh || exit 1;

nnet1=exp/scores/r100-v4
nnet2=../v0.1/exp/scores/r100-v4
be1=be_v3/cosine_cal_v1_sre21
be1s=be_v3/cosine_snorm1000_v1_cal_v1_sre21
be2=be_v4_thrahc0.8/cosine_cal_v1_sre21
be2s=be_v4_thrahc0.8/cosine_snorm1000_v1_cal_v1_sre21
be3=be_v6_selfatt_a2/cosine_cal_v1_sre21
be3s=be_v6_selfatt_a2/cosine_snorm1000_v1_cal_v1_sre21
be4=be_v7_selfatt_a2_att_a4/cosine_cal_v1_sre21
be4s=be_v7_selfatt_a2_att_a4/cosine_snorm1000_v1_cal_v1_sre21

echo "This is just a fusion example, \
     you won't be able to run it if you don't have all the systems need for the fusion"

system_names="mxnet-b1 mxnet-b2 mxnet-b3 mxnet-b4 mxnet-b1s mxnet-b2s mxnet-b3s mxnet-b4s pt-b1 pt-b2 pt-b3 pt-b4 pt-b1s pt-b2s pt-b3s pt-b4s"
system_dirs="$nnet1/$be1 $nnet1/$be2 $nnet1/$be3 $nnet1/$be4 \
$nnet1/$be1s $nnet1/$be2s $nnet1/$be3s $nnet1/$be4s \
$nnet2/$be1 $nnet2/$be2 $nnet2/$be3 $nnet2/$be4 \
$nnet2/$be1s $nnet2/$be2s $nnet2/$be3s $nnet2/$be4s"

output_dir=exp/fusion/v1.1_ptrn${p_trn}_l2${fus_l2_reg}

if [ $stage -le 1 ];then
    local/fusion_sre21av_v1.sh --cmd "$train_cmd --mem 24G" \
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
      local/score_sre21_official.sh $sre21_dev_root visual dev $output_dir/$i
      local/score_sre21_official.sh $sre21_eval_root visual eval $output_dir/$i 
    fi
  done
fi

