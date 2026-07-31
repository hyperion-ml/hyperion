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

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

list_dir=data/train_lid_proc_audio_no_sil

if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

lid_nnet_dir=exp/lid_nnets/lresnet34_lid_v1
# Network Training
if [ $stage -le 1 ]; then

  mkdir -p $lid_nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $lid_nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    train_xvector_from_wav.py resnet \
    --cfg conf/train_lresnet34_lid_v1.yaml \
    --data.train.dataset.recordings-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_train_lid/train.scp \
    --data.train.dataset.class-file $list_dir/lists_train_lid/class2int \
    --data.val.dataset.recordings-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_train_lid/val.scp \
    --trainer.exp-path $lid_nnet_dir $extra_args \
    --num-gpus $ngpu
fi

