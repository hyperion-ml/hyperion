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
num_workers=8

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

batch_size=$(($advft_batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $advft_eff_batch_size | awk '{ print int($2/$1+0.5)}')
log_interval=$(echo 100*$grad_acc_steps | bc)
list_dir=data/${nnet_data}_proc_audio_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
  mkdir -p $advft_nnet_dir/log
  $cuda_cmd --gpu $ngpu $advft_nnet_dir/log/train.log \
      hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
      torch-adv-finetune-xvec-from-wav.py --feats $feat_config $aug_opt \
      --audio-path $list_dir/wav.scp \
      --time-durs-file $list_dir/utt2dur \
      --train-list $list_dir/lists_xvec/train.scp \
      --val-list $list_dir/lists_xvec/val.scp \
      --class-file $list_dir/lists_xvec/class2int \
      --min-chunk-length $min_chunk --max-chunk-length $max_chunk \
      --iters-per-epoch $ipe \
      --batch-size $batch_size \
      --num-workers $num_workers \
      --grad-acc-steps $grad_acc_steps $advft_opt_opt $advft_lrs_opt \
      --epochs $advft_nnet_num_epochs \
      --s $s --margin $advft_margin --margin-warmup-epochs $advft_margin_warmup \
      --num-gpus $ngpu \
      --train-mode ft-full \
      --log-interval $log_interval \
      --in-model-path $nnet \
      --exp-path $advft_nnet_dir $advft_attack_opts $args

fi
#

exit
