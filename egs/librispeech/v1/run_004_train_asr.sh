#!/bin/bash
# Copyright
#                2022   Johns Hopkins University (Author: Yen-Ju Lu)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=2
config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

train_dir=data/${nnet_train_data}
val_dir=data/${nnet_val_data}

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then

  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s1_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-wav2rnn-transducer $nnet_type \
    --cfg $nnet_s1_cfg \
    --data.train.dataset.recordings-file $train_dir/recordings.csv \
    --data.train.dataset.segments-file $train_dir/segments.csv \
    --data.train.dataset.bpe-model $token_model \
    --data.val.dataset.recordings-file $val_dir/recordings.csv \
    --data.val.dataset.segments-file $val_dir/segments.csv \
    --trainer.exp-path $nnet_s1_dir $args \
    --num-gpus $ngpu

fi

