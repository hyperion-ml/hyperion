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
exp=xvector_nnets/baseline/fbank80_stmn_ecapatdnn512x3.v3.0.s1_voxceleb2cat_500
model=exp/xvector_nnets/baseline/fbank80_stmn_ecapatdnn512x3.v3.0.s1_voxceleb2cat_500/model_ep0070.pth

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

# Network Training
if [ $stage -le 1 ]; then

  mkdir -p $nnet_s1_dir/log
  #mkdir -p exp/check_bug/log
  $cuda_cmd \
    --gpu $ngpu $exp/log/train.log  \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-wav2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $exp \
    --num-gpus $ngpu \

fi


# Large Margin Fine-tuning
if [ $stage -le 2 ]; then
 if [ "$use_wandb" == "true" ];then
   extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
 fi
 mkdir -p $exp/finetune/log
 $cuda_cmd \
   --gpu $ngpu $exp/finetune/log/train.log \
   hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
   hyperion-finetune-wav2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s2_args $extra_args \
   --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
   --data.train.dataset.segments-file $train_data_dir/segments.csv \
   --data.train.dataset.class-files $train_data_dir/speaker.csv \
   --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
   --data.val.dataset.segments-file $val_data_dir/segments.csv \
   --in-model-file $model \
   --trainer.exp-path $exp \
   --num-gpus $ngpu \

fi