#!/bin/bash
# Copyright     2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

net_type=ffnet
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
    #$cuda_cmd $exp_dir/main.log \
    qsub -e err.log -o o.log -cwd -l 'hostname=[bc][1]*,gpu=1' \
	      CUDA_VISIBLE_DEVICES=`free-gpu` \
	      $MYTORCH local/main.py \
	      --batch-size $batch_size \
	      --lr $lr --momentum $momentum

    
    
    source deactivate $TORCH
fi

