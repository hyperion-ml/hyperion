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

batch_size=$(($spknet_batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $spknet_eff_batch_size | awk '{ print int($2/$1+0.5)}')
log_interval=$(echo 100*$grad_acc_steps | bc)
list_dir=data/${spknet_data}_proc_audio_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then

    mkdir -p $spknet_dir/log
    $cuda_cmd --gpu $ngpu $spknet_dir/log/train.log \
	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	torch-train-xvec-from-wav.py  $spknet_command --cfg $spknet_config \
	--audio-path $list_dir/wav.scp \
	--time-durs-file $list_dir/utt2dur \
	--train-list $list_dir/lists_xvec/train.scp \
	--val-list $list_dir/lists_xvec/val.scp \
	--class-file $list_dir/lists_xvec/class2int \
	--batch-size $batch_size \
	--num-workers $num_workers \
	--grad-acc-steps $grad_acc_steps \
	--num-gpus $ngpu \
	--log-interval $log_interval \
	--exp-path $spknet_dir $args

fi
