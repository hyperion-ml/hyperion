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
resume=false
interactive=false
num_workers=8
lid_ipe=1
. parse_options.sh || exit 1;
. $config_file
. datapath.sh

list_dir=data/train_lid_proc_audio_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

lid_nnet_dir=exp/lid_nnets/lresnet34_lid_v1
# Network Training
if [ $stage -le 1 ]; then

  train_exec=torch-train-resnet-xvec-from-wav.py
  mkdir -p $lid_nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $lid_nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    $train_exec --cfg conf/lresnet34_lid_v1.yaml \
    --audio-path $list_dir/wav.scp \
    --time-durs-file $list_dir/utt2dur \
    --train-list $list_dir/lists_train_lid/train.scp \
    --val-list $list_dir/lists_train_lid/val.scp \
    --class-file $list_dir/lists_train_lid/class2int \
    --iters-per-epoch $lid_ipe \
    --num-workers $num_workers \
    --num-gpus $ngpu \
    --exp-path $lid_nnet_dir $args

fi

exit
