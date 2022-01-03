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

batch_size=$(($ft2_batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $ft2_eff_batch_size | awk '{ print int($2/$1)}')
log_interval=$(echo 100*$grad_acc_steps | bc)
list_dir=data/${nnet_adapt_data}_proc_audio_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
  mkdir -p $ft2_nnet_dir/log
  $cuda_cmd --gpu $ngpu $ft2_nnet_dir/log/train.log \
      hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
      torch-finetune-xvec-dfr-from-wav.py --feats $feat_config $aug_opt \
      --audio-path $list_dir/wav.scp \
      --time-durs-file $list_dir/utt2dur \
      --train-list $list_dir/lists_xvec/train.scp \
      --val-list $list_dir/lists_xvec/val.scp \
      --class-file $list_dir/lists_xvec/class2int \
      --min-chunk-length $ft2_min_chunk --max-chunk-length $ft2_max_chunk \
      --iters-per-epoch $ft2_ipe \
      --batch-size $batch_size \
      --num-workers $num_workers $ft2_opt_opt $ft2_lrs_opt \
      --grad-acc-steps $grad_acc_steps \
      --reg-layers-classif $reg_layers_classif \
      --reg-weight-classif $ft2_reg_weight_embed \
      --epochs $ft2_nnet_num_epochs \
      --cos-scale $s --margin $margin --margin-warmup-epochs $ft2_margin_warmup \
      --num-gpus $ngpu \
      --log-interval $log_interval \
      --in-model-path $ft_nnet \
      --train-mode ft-embed-affine \
      --exp-path $ft2_nnet_dir $args

fi
#

exit
