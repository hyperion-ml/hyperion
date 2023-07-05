#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=0
config_file=default_config.sh
use_gpu=false
nnet_stage=1
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
  lid_args="--use-gpu true"
  lid_cmd="$cuda_eval_cmd --mem 6G"
else
  lid_cmd="$train_cmd --mem 12G"
fi

if [ $nnet_stage -eq 1 ];then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ];then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
elif [ $nnet_stage -eq 3 ];then
  nnet=$nnet_s3
  nnet_name=$nnet_s3_name
fi

lid_dir=exp/resnet1d/$nnet_name

rm -f $lid_dir/overall_lid_score.txt

# Extracts x-vectors for evaluation
for name in $test_data  # $dev_data $test_data 
  do
    nj=40
    steps_lid/identificate_wav2vec2resnet1d.sh \
      --cmd "$lid_cmd" --nj $nj ${lid_args} \
      $nnet data/$name \
      $lid_dir/$name data/$nnet_data/langs
  done

exit
