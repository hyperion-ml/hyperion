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
resume=false
interactive=false
num_workers=8

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

batch_size=$(($batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $eff_batch_size | awk '{ print int($2/$1+0.5)}')
log_interval=$(echo 100*$grad_acc_steps | bc)
list_dir=data/${nnet_data}_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

train_exec=torch-train-${narch}-dvae.py

# Network Training
if [ $stage -le 1 ]; then
  mkdir -p $nnet_dir/log
  $cuda_cmd --gpu $ngpu $nnet_dir/log/train.log \
      hyp_utils/torch.sh --num-gpus $ngpu \
      $train_exec \
      --data-rspec scp:$list_dir/feats.scp \
      --train-list $list_dir/lists_xvec/train.scp \
      --train-pair-list $list_dir/lists_xvec/augm2clean.scp \
      --val-list $list_dir/lists_xvec/val.scp \
      --val-pair-list $list_dir/lists_xvec/augm2clean.scp \
      --num-frames-file $list_dir/utt2num_frames \
      --min-chunk-length $min_chunk --max-chunk-length $max_chunk \
      --iters-per-epoch $ipe \
      --batch-size $batch_size \
      --num-workers $num_workers $opt_opt $lrs_opt \
      --grad-acc-steps $grad_acc_steps \
      --epochs $nnet_num_epochs \
      --z-dim $latent_dim $enc_opt $dec_opt $vae_opt \
      --num-gpus $ngpu \
      --log-interval $log_interval \
      --exp-path $nnet_dir $args

fi


exit
