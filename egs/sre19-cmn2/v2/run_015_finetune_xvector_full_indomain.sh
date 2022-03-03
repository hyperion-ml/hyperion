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

batch_size=$(($ft3_batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $ft3_eff_batch_size | awk '{ print int($2/$1)}')
log_interval=$(echo 100*$grad_acc_steps | bc)

list_dir=data/${nnet_adapt_data}_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
    mkdir -p $ft3_nnet_dir/log
    $cuda_cmd --gpu $ngpu $ft3_nnet_dir/log/train.log \
      hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
      torch-finetune-xvec-dfr.py \
      --data-rspec scp:$list_dir/feats.scp \
      --train-list $list_dir/lists_xvec/train.scp \
      --val-list $list_dir/lists_xvec/val.scp \
      --class-file $list_dir/lists_xvec/class2int \
      --num-frames-file $list_dir/utt2num_frames \
      --min-chunk-length $ft3_min_chunk --max-chunk-length $ft3_max_chunk \
      --iters-per-epoch $ft3_ipe \
      --batch-size $batch_size \
      --num-workers $num_workers $ft3_opt_opt $ft3_lrs_opt \
      --grad-acc-steps $grad_acc_steps \
      --reg-layers-classif 0 \
      --reg-weight-classif $ft3_reg_weight_embed \
      --reg-layers-enc 0 1 2 3 4 \
      --reg-weight-enc $ft3_reg_weight_enc \
      --epochs $ft3_nnet_num_epochs \
      --cos-scale $s --margin $margin --margin-warmup-epochs $ft3_margin_warmup \
      --num-gpus $ngpu \
      --log-interval $log_interval \
      --in-model-path $ft2_nnet \
      --train-mode ft-full \
      --exp-path $ft3_nnet_dir $args

fi
#

exit
