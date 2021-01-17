#!/bin/bash
# Copyright     2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e


net_type=resnet34
resnet_args="--in-stride 1 --no-maxpool"
num_gpus=2
lr=0.1
batch_size=128
momentum=0.9
stage=1
. parse_options.sh || exit 1;

net_name=$net_type
exp_dir=./exp/mnist_${net_name}
mkdir -p $exp_dir

#Train network
if [ $stage -le 1 ]; then
    mkdir -p $exp_dir/log
    $cuda_cmd --gpu $num_gpus $exp_dir/log/train.log \
        hyp_utils/torch.sh --num-gpus $num_gpus \
	local/resnet-mnist.py \
	--resnet-type $net_type $resnet_args \
	--batch-size $batch_size \
	--opt-optimizer sgd \
	--num-gpus $num_gpus \
	--opt-lr $lr --opt-momentum $momentum --opt-weight-decay 5e-4 \
	--lrsch-lrsch-type exp_lr \
	--lrsch-decay-rate 0.1 \
	--lrsch-decay-steps 50000 \
	--lrsch-hold-steps 20000 \
	--lrsch-min-lr 0.001 \
	--lrsch-warmup-steps 5000 \
	--lrsch-update-lr-on-opt-step \
	--epochs 300 \
	--exp-path $exp_dir
    
fi


