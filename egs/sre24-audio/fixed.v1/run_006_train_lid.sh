#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=4
config_file=default_config.sh
interactive=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

train_data_dir=data/${nnet_data}_lid_train
val_data_dir=data/${nnet_data}_lid_val

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

nnet_s1_cfg=$lid_nnet_s1_cfg
nnet_s1_dir=$lid_nnet_s1_dir
nnet_s1=$lid_nnet_s1
nnet_type=$lid_nnet_type

# Network Training
if [ $stage -le 1 ]; then
  
  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s1_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-wav2xvector $nnet_type --cfg $nnet_s1_cfg \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/language.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $nnet_s1_dir \
    --num-gpus $ngpu \
  
fi

