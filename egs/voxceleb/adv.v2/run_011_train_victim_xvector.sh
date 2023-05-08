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
num_workers=""
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

nnet_type=$spknet_command
nnet_data=$spknet_data
nnet_dir=$spknet_dir
nnet_cfg=$spknet_config
list_dir=data/${nnet_data}_proc_audio_no_sil

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

nnet_type=$spknet_command
nnet_dir

# Network Training
if [ $stage -le 1 ]; then
  
  mkdir -p $nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    train_xvector_from_wav.py $nnet_type --cfg $nnet_cfg \
    --data.train.dataset.recordings-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.recordings-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --trainer.exp-path $nnet_dir \
    --num-gpus $ngpu
  
fi

