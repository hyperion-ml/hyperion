#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
use_a100=true   # Set to true to request A100; false otherwise

if [ "$use_a100" == "true" ]; then
  export cuda_cmd="slurm.pl --gpu 1 --opt '--partition=gpu-a100 --account=a100acct'"
  ngpu=1
else
  export cuda_cmd="slurm.pl --gpu 4 --opt '--partition=gpu'"
  ngpu=4
fi

config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh


# train_data_dir=data/${full_dataset}_xvector_train
# val_data_dir=data/${full_dataset}_xvector_val


train_data_dir=data/${full_dataset}_xvector_train
val_data_dir=data/${full_dataset}_xvector_val

alpha=1
position=-1
pourcentage_poisoned=20
n_attacks=20
n_speakers=250
version=norm_single_target
trigger_dir=data/triggers/click/attack_$n_attacks/norm
attack_dir=exp/multitarget/attack_${n_attacks}_${version}
attack_infos=$attack_dir/infos.csv

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
    export cuda_cmd=run.pl
fi


# if [ $stage -le 1 ];then
#   mkdir -p $attack_dir
#   hyperion-dataset create_attacks\
#                    --n-attacks $n_attacks \
#                    --n-speakers $n_speakers \
#                    --full-dataset $train_data_dir \
#                    --pourcentage-poisoned 0.${pourcentage_poisoned} \
#                    --trigger-dir $trigger_dir \
#                    --attack-dir $attack_dir \
#                    --joint-classes speaker --min-train-samples 5 \
#                    --seed 1123581322 
# fi

#Network Training
if [ $stage -le 2 ]; then
  mkdir -p $attack_dir/log
  $cuda_cmd \
    --gpu $ngpu $attack_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-multi-poisoned $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $attack_dir \
    --num-gpus $ngpu \
    --n-attacks $n_attacks \
    --attack-infos $attack_infos \
    --alpha-min $alpha\
    --alpha-max $alpha\
    --trigger-position $position

fi
