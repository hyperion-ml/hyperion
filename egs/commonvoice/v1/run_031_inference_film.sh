#!/bin/bash
# Copyright
#                2022   Johns Hopkins University (Author: Yen-Ju Lu)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh
use_gpu=false
nnet_stage=1
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
  transducer_args="--use-gpu true"
  transducer_cmd="$cuda_eval_cmd --mem 6G"
else
  transducer_cmd="$train_cmd --mem 12G"
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

transducer_dir=exp/transducer/$nnet_name


# test_data=test_clean


# Extracts x-vectors for evaluation
for name in $test_data
do
  nj=40
  steps_transducer/decode_wav2vec2rnn_film_transducer.sh \
      --cmd "$transducer_cmd --mem 12G" --nj $nj ${transducer_args} \
      $nnet data/$name \
      $transducer_dir/$name $bpe_model data/$nnet_data/langs
done

