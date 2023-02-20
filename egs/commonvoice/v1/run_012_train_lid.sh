#!/bin/bash
# Copyright
#                2022   Johns Hopkins University (Author: Yen-Ju Lu)
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
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

train_dir=data/${nnet_data}/
val_dir=data/${dev_data}/

#add extra args from the command line arguments
if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
    extra_args="--data.val.data_loader.num-workers $num_workers"
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
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu --max-split-size-mb 512 \
    train_wav2vec2languageid.py $nnet_type \
    --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.audio-file $train_dir/wav.scp \
    --data.train.dataset.segments-file $train_dir/utt2spk \
    --data.train.dataset.class-names "language" \
    --data.train.dataset.class-files $train_dir/langs \
    --data.train.dataset.language-id-file $train_dir/utt2lang \
    --data.train.dataset.bpe-model $bpe_model \
    --data.train.dataset.text-file $train_dir/text \
    --data.val.dataset.audio-file $val_dir/wav.scp \
    --data.val.dataset.segments-file $val_dir/utt2spk \
    --data.val.dataset.class-names "language" \
    --data.val.dataset.class-files $val_dir/langs \
    --data.val.dataset.language-id-file $val_dir/utt2lang \
    --data.val.dataset.text-file $val_dir/text \
    --trainer.exp-path $nnet_s1_dir $args \
    --data.train.dataset.time-durs-file $train_dir/utt2dur \
    --data.val.dataset.time-durs-file $val_dir/utt2dur \
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
    finetune_wav2vec2languageid.py $nnet_type \
    --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
    --data.train.dataset.audio-file $train_dir/wav.scp \
    --data.train.dataset.segments-file $train_dir/utt2spk \
    --data.train.dataset.class-names "language" \
    --data.train.dataset.class-files $train_dir/langs \
    --data.train.dataset.language-id-file $train_dir/utt2lang \
    --data.train.dataset.bpe-model $bpe_model \
    --data.train.dataset.text-file $train_dir/text \
    --data.val.dataset.audio-file $val_dir/wav.scp \
    --data.val.dataset.segments-file $val_dir/utt2spk \
    --data.val.dataset.class-names "language" \
    --data.val.dataset.class-files $val_dir/langs \
    --data.val.dataset.language-id-file $val_dir/utt2lang \
    --data.val.dataset.text-file $val_dir/text \
    --trainer.exp-path $nnet_s2_dir $args \
    --in-model-file $nnet_s1 \
    --data.train.dataset.time-durs-file $train_dir/utt2dur \
    --data.val.dataset.time-durs-file $val_dir/utt2dur \
    --num-gpus $ngpu
  
fi

if [ $stage -le 3 ]; then

  if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.wandb.name $nnet_s3_name.$(date -Iminutes)"
  fi
  

  mkdir -p $nnet_s3_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s3_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    finetune_wav2vec2languageid.py $nnet_type \
    --cfg $nnet_s3_base_cfg $nnet_s3_args $extra_args \
    --data.train.dataset.audio-file $train_dir/wav.scp \
    --data.train.dataset.segments-file $train_dir/utt2spk \
    --data.train.dataset.class-names "language" \
    --data.train.dataset.class-files $train_dir/langs \
    --data.train.dataset.language-id-file $train_dir/utt2lang \
    --data.train.dataset.bpe-model $bpe_model \
    --data.train.dataset.text-file $train_dir/text \
    --data.val.dataset.audio-file $val_dir/wav.scp \
    --data.val.dataset.segments-file $val_dir/utt2spk \
    --data.val.dataset.class-names "language" \
    --data.val.dataset.class-files $val_dir/langs \
    --data.val.dataset.language-id-file $val_dir/utt2lang \
    --data.val.dataset.text-file $val_dir/text \
    --trainer.exp-path $nnet_s3_dir $args \
    --in-model-file $nnet_s2 \
    --data.train.dataset.time-durs-file $train_dir/utt2dur \
    --data.val.dataset.time-durs-file $val_dir/utt2dur \
    --num-gpus $ngpu
fi

