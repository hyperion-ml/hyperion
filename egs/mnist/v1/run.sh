#!/bin/bash
# Copyright     2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

net_type=fcnet
lr=0.01
batch_size=64
momentum=0.5
stage=1
. parse_options.sh || exit 1;

net_name=$net_type
exp_dir=./exp/$net_name
mkdir -p $exp_dir

#Train embeddings
if [ $stage -le 1 ]; then
    #$cuda_cmd $exp_dir/main.log \
    qsub -e err.log -o o.log -cwd -l 'hostname=[bc][1][123456789]*,gpu=1' \
         hyp_utils/torch.sh --num-gpus 1 \
	 local/main.py \
	     --net-type $net_type \
	     --batch-size $batch_size \
	     --opt-optimizer sgd \
	     --num-gpus 1 \
	     --opt-lr $lr --opt-momentum $momentum \
	     --lrsch-lrsch-type exp_lr \
	     --lrsch-decay-rate 0.1 \
	     --lrsch-decay-steps 5 \
	     --lrsch-hold-steps 3 \
	     --lrsch-min-lr 0.001 \
	     --lrsch-warmup-steps 100 \
	     --lrsch-t 500 \
	     --lrsch-t-mul 2 \
	     --lrsch-gamma 0.9 \
	     --lrsch-warm-restarts \
	     --lrsch-update-lr-on-batch \
	     --lrsch-patience 1 \
	     --lrsch-threshold 1e-1 \
	     --resume \
	     --epochs 1 \
	     --exp-path $exp_dir
    
fi

