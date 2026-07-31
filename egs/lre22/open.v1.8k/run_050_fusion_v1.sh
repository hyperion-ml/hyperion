#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

score_dir_0=exp/scores
nnet_1=fbank64_stmn_ecapatdnn2048x4_v1.0.s1
nnet_2=fbank64_stmn_fwseres2net50s8_v1.0.s1
be_1=pca1_cw_lnorm_lgbe_lre22_aug
score_dirs="$score_dir_0/$nnet_1/$be_1
$score_dir_0/$nnet_2/$be_1"

train_score_dirs=$(echo $score_dirs | awk '{ for(i=1;i<=NF;i++){ $i=$i"_p12/cal_v1" }; print $0}')
test_score_dirs=$(echo $score_dirs | awk '{ for(i=1;i<=NF;i++){ $i=$i"/cal_v1" }; print $0}')

output_dir=exp/fusion/fus_v1.0

local/train_fusion_lre22.sh "$train_score_dirs" $output_dir/train
local/score_lre22.sh \
  dev \
  ${output_dir}/train/lre22_dev_scores.tsv \
  ${output_dir}/train/lre22_dev_results

local/eval_fusion_lre22.sh "$test_score_dirs" $output_dir/train/fus.mat $output_dir/test

local/score_lre22.sh \
  dev \
  ${output_dir}/test/lre22_dev_scores.tsv \
  ${output_dir}/test/lre22_dev_results

local/score_lre22.sh eval \
  ${output_dir}/test/lre22_eval_scores.tsv \
  ${output_dir}/test/lre22_eval_results





		   
