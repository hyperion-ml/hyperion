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
resume=false
interactive=false
num_workers=3

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

batch_size=$(($ft_batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $ft_eff_batch_size | awk '{ print int($2/$1)}')
log_interval=$(echo 100*$grad_acc_steps | bc)

list_dir=data/${nnet_data}_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
  mkdir -p $ft_nnet_dir/log
  $cuda_cmd --gpu $ngpu $ft_nnet_dir/log/train.log \
      hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
      torch-finetune-xvec.py \
      --data-rspec scp:$list_dir/feats.scp \
      --train-list $list_dir/lists_xvec/train.scp \
      --val-list $list_dir/lists_xvec/val.scp \
      --class-file $list_dir/lists_xvec/class2int \
      --num-frames-file $list_dir/utt2num_frames \
      --min-chunk-length $ft_min_chunk --max-chunk-length $ft_max_chunk \
      --iters-per-epoch $ft_ipe \
      --batch-size $batch_size \
      --num-workers $num_workers $ft_opt_opt $ft_lrs_opt \
      --grad-acc-steps $grad_acc_steps \
      --epochs $ft_nnet_num_epochs \
      --cos-scale $s --margin $margin --margin-warmup-epochs $ft_margin_warmup \
      --num-gpus $ngpu \
      --log-interval $log_interval \
      --in-model-path $nnet \
      --train-mode ft-full \
      --exp-path $ft_nnet_dir $args
fi
#

exit
