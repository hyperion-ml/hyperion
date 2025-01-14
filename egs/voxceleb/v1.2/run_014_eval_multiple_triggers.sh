#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=1
config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_gpu=true
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 12G"
  num_gpus=0
fi


train_data_dir=data/${nnet_data}_xvector_train
test_data_dir=data/${nnet_data}_xvector_test
dataset=500


alpha=rand
position=-1
pourcentage_poisoned=15
n_attacks=20
version=1.2
attack_dir=exp/attack_${n_attacks}_clusters_$version
attack_infos=$attack_dir/infos.csv
model=ep0100
model_path=$attack_dir/model_$model.pth

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
    export cuda_cmd=run.pl
fi



if [ $stage -le 1 ]; then
  mkdir -p $attack_dir/log
  $cuda_cmd \
    --gpu $ngpu $attack_dir/log/eval_$model.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-eval-wav2xvector-poi-multi $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $test_data_dir/recordings.csv \
    --data.val.dataset.segments-file $test_data_dir/segments.csv \
    --num-gpus $ngpu \
    --model-path $model_path \
    --n-attacks $n_attacks \
    --attack-infos $attack_infos \
    --alpha-min $alpha_min\
    --alpha-max $alpha_max\
    --trigger-position $position \
    --exp-path $attack_dir

fi
