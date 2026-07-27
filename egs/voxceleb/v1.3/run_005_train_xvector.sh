#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
# ---------------------------------------------------------------------------
# run_005_train_xvector.sh
# ---------------------------------------------------------------------------
# Stage 5 of the VoxCeleb v1.2 recipe. It trains the x-vector/ECAPA models
# (stage 1) and optionally fine-tunes them with large-margin objectives
# (stage 2). Toggle `stage`, `ngpu`, `use_tb`, `use_wandb`, etc. to match
# your environment and logging preferences.
# ---------------------------------------------------------------------------

#
. ./cmd.sh
. ./path.sh
set -e

stage=1           # stage threshold to resume pipeline
ngpu=4            # number of GPUs used per training command
config_file=default_config.sh  # Neural network config to source
interactive=false  # Use hyperion-submit local instead of Slurm (useful for debugging)
num_workers=""   # Override data loader workers when set
use_tb=false      # Enable TensorBoard logging when true
use_wandb=false   # Enable Weights & Biases logging when true

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
    export cuda_cmd="hyperion-submit local"
fi

# Network Training
if [ $stage -le 1 ]; then
  echo "[run005][stage1] Training base x-vector model: $nnet_name"
  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --num-gpus $ngpu --output-file $nnet_s1_dir/log/train.log -- \
    hyperion-train-wav2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $nnet_s1_dir \
    --num-gpus $ngpu \
  
fi


# Large Margin Fine-tuning
if [ $stage -le 2 ]; then
  if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
  fi
  echo "[run005][stage2] Fine-tuning with large-margin objective: $nnet_s2_name"
  mkdir -p $nnet_s2_dir/log
  $cuda_cmd \
    --num-gpus $ngpu --output-file $nnet_s2_dir/log/train.log -- \
    hyperion-finetune-wav2xvector $nnet_type --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --in-model-file $nnet_s1 \
    --trainer.exp-path $nnet_s2_dir \
    --num-gpus $ngpu \
  
fi
