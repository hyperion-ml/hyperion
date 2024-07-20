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

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

train_data_dir=data/${nnet_train_data}
val_data_dir=data/${nnet_val_data}

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
  
  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s1_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-wav2vec2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/spoof_det.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $nnet_s1_dir \
    --num-gpus $ngpu 
  
fi

# Finetune full model
if [ $stage -le 2 ]; then
  mkdir -p $nnet_s2_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s2_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-finetune-wav2vec2xvector $nnet_type --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/spoof_det.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --in-model-file $nnet_s1 \
    --trainer.exp-path $nnet_s2_dir \
    --num-gpus $ngpu 
  
fi
exit
  
# Finetune full model
if [ $stage -le 3 ]; then
  mkdir -p $nnet_s3_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s3_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-finetune-wav2vec2xvector $nnet_type --cfg $nnet_s3_base_cfg $nnet_s3_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/spoof_det.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --in-model-file $nnet_s2 \
    --trainer.exp-path $nnet_s3_dir \
    --num-gpus $ngpu \
  
fi

exit
# Network Training
if [ $stage -le 1 ]; then
  
  mkdir -p $nnet_s1_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s1_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-wav2xvector $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/spoof_det.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $nnet_s1_dir \
    --num-gpus $ngpu
  
fi


# # Large Margin Fine-tuning
# if [ $stage -le 2 ]; then
#   if [ "$use_wandb" == "true" ];then
#     extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
#   fi
#   mkdir -p $nnet_s2_dir/log
#   $cuda_cmd \
#     --gpu $ngpu $nnet_s2_dir/log/train.log \
#     hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
#     hyperion-finetune-wav2xvector $nnet_type --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
#     --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
#     --data.train.dataset.segments-file $train_data_dir/segments.csv \
#     --data.train.dataset.class-files $train_data_dir/speaker.csv \
#     --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
#     --data.val.dataset.segments-file $val_data_dir/segments.csv \
#     --in-model-file $nnet_s1 \
#     --trainer.exp-path $nnet_s2_dir \
#     --num-gpus $ngpu \
  
# fi
