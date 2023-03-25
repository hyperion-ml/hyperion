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

list_dir=data/${nnet_data}_proc_audio_no_sil

#add extra args from the command line arguments
if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi
if [ "$use_tb" == "true" ];then
    extra_args="$extra_args --trainer.use-tensorboard"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

if [ "$use_wandb" == "true" ];then
  extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project voxceleb-v2 --trainer.wandb.name $nnet_s1_name.$(date -Iminutes)"
fi


# Network Training
if [ $stage -le 1 ]; then

  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s1_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    train_wav2vec2xvector.py $nnet_type \
    --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.audio-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.audio-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --trainer.exp-path $nnet_s1_dir $args \
    --num-gpus $ngpu
  
fi

if [ $stage -le 2 ]; then

  if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
  fi
  
  mkdir -p $nnet_s2_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s2_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    finetune_wav2vec2xvector.py $nnet_type \
    --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
    --data.train.dataset.audio-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.audio-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --in-model-file $nnet_s1 \
    --trainer.exp-path $nnet_s2_dir $args \
    --num-gpus $ngpu \
  
fi

if [ $stage -le 3 ]; then

  if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.wandb.name $nnet_s3_name.$(date -Iminutes)"
  fi
  
  mkdir -p $nnet_s3_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s3_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    finetune_wav2vec2xvector.py $nnet_type \
    --cfg $nnet_s3_base_cfg $nnet_s3_args $extra_args \
    --data.train.dataset.audio-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.audio-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --in-model-file $nnet_s2 \
    --trainer.exp-path $nnet_s3_dir $args \
    --num-gpus $ngpu \
  
fi

