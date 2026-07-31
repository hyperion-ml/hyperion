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

train_data_dir=data/${nnet_data}_xvector_train
val_data_dir=data/${nnet_data}_xvector_val

#add extra args from the command line arguments
if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi
if [ "$use_tb" == "true" ];then
    extra_args="$extra_args --trainer.use-tensorboard"
fi
if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project voxceleb-v1.1 --trainer.wandb.name $nnet_name.$(date -Iminutes)"
fi

if [ "$interactive" == "true" ];then
    echo "[run005] Interactive mode: using hyperion-submit local"
    export cuda_cmd="hyperion-submit local"
fi

# Network Training
if [ $stage -le 1 ]; then
  echo "[run005][stage1] Training base wav2vec2 x-vector model: $nnet_s1_name"
  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --num-gpus $ngpu --output-file $nnet_s1_dir/log/train.log -- \
    hyperion-train-wav2vec2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $nnet_s1_dir \
    --num-gpus $ngpu \
  
fi


# Finetune full model
if [ $stage -le 2 ]; then
  if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
  fi
  echo "[run005][stage2] Fine-tuning wav2vec2 x-vector model: $nnet_s2_name"
  mkdir -p $nnet_s2_dir/log
  $cuda_cmd \
    --num-gpus $ngpu --output-file $nnet_s2_dir/log/train.log -- \
    hyperion-finetune-wav2vec2xvector $nnet_type --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --in-model-file $nnet_s1 \
    --trainer.exp-path $nnet_s2_dir \
    --num-gpus $ngpu \
  
fi

# Finetune full model
if [ $stage -le 3 ]; then
  if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.wandb.name $nnet_s3_name.$(date -Iminutes)"
  fi
  echo "[run005][stage3] Large-margin fine-tuning: $nnet_s3_name"
  mkdir -p $nnet_s3_dir/log
  $cuda_cmd \
    --num-gpus $ngpu --output-file $nnet_s3_dir/log/train.log -- \
    hyperion-finetune-wav2vec2xvector $nnet_type --cfg $nnet_s3_base_cfg $nnet_s3_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --in-model-file $nnet_s2 \
    --trainer.exp-path $nnet_s3_dir \
    --num-gpus $ngpu \
  
fi
