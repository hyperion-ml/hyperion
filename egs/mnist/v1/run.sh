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
    source activate $TORCH
    MYTORCH=$(which python)
    #echo qsub -e err.log -o o.log -cwd -l 'hostname=[bc][1]*,gpu=1' \
    #$cuda_cmd $exp_dir/main.log \
    $MYTORCH local/main.py --no-cuda \
	     --net-type $net_type \
	     --batch-size $batch_size \
	     --opt-optimizer sgd \
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
	     --exp-path $exp_dir
    
    source deactivate $TORCH
fi

