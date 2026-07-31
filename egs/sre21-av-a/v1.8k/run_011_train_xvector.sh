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

list_dir=data/${nnet_data}_proc_audio_no_sil

if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
  
  mkdir -p $nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    train_xvector_from_wav.py $nnet_type \
    --cfg $nnet_base_cfg $nnet_args $extra_args \
    --data.train.dataset.recordings-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.recordings-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --trainer.exp-path $nnet_dir \
    --num-gpus $ngpu \
  
fi

# Large Margin Fine-tuning
if [ $stage -le 2 ]; then
  mkdir -p $ft_nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $ft_nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    finetune_xvector_from_wav.py $nnet_type \
    --cfg $ft_nnet_base_cfg $ft_nnet_args $extra_args \
    --data.train.dataset.recordings-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.recordings-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --in-model-file $nnet \
    --trainer.exp-path $ft_nnet_dir \
    --num-gpus $ngpu \
  
fi
