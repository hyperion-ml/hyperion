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
interactive=false
num_workers=""
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

list_dir=data/${nnet_data}_proc_audio_no_sil

#add extra args from the command line arguments
if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi
if [ "$use_tb" == "true" ];then
    extra_args="$extra_args --trainer.use-tensorboard"
fi
if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project lre22-fixed-v1.8k --trainer.wandb.name $nnet_s1_name.$(date -Iminutes)"
fi
if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
    mkdir -p $nnet_s1_dir/log
    if [ ! -f "$nnet_s0" ];then
	$cuda_cmd \
	    --gpu $ngpu $nnet_s1_dir/log/train.log \
	    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	    train_xvector_from_wav.py $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
	    --data.train.dataset.recordings-file $list_dir/wav.scp \
	    --data.train.dataset.segments-file $list_dir/train_val_split/train_segments.csv \
	    --data.train.dataset.class-files $list_dir/train_val_split/class_file.csv \
	    --data.val.dataset.recordings-file $list_dir/wav.scp \
	    --data.val.dataset.segments-file $list_dir/train_val_split/val_segments.csv \
	    --trainer.exp-path $nnet_s1_dir \
	    --num-gpus $ngpu #--master-port 3456
    else
	$cuda_cmd \
	    --gpu $ngpu $nnet_s1_dir/log/train.log \
	    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	    finetune_xvector_from_wav.py $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
	    --data.train.dataset.recordings-file $list_dir/wav.scp \
	    --data.train.dataset.segments-file $list_dir/train_val_split/train_segments.csv \
	    --data.train.dataset.class-files $list_dir/train_val_split/class_file.csv \
	    --data.val.dataset.recordings-file $list_dir/wav.scp \
	    --data.val.dataset.segments-file $list_dir/train_val_split/val_segments.csv \
	    --in-model-file $nnet_s0 \
	    --trainer.exp-path $nnet_s1_dir \
	    --num-gpus $ngpu 

    fi
    
fi


# Class-balanced Fine-tuning
if [ $stage -le 2 ] && [ $max_stage -le 2 ]; then
    if [ "$use_wandb" == "true" ];then
	extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
    fi
    mkdir -p $nnet_s2_dir/log
    $cuda_cmd \
	--gpu $ngpu $nnet_s2_dir/log/train.log \
	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	finetune_xvector_from_wav.py $nnet_type --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
	--data.train.dataset.recordings-file $list_dir/wav.scp \
	--data.train.dataset.segments-file $list_dir/train_val_split/train_segments.csv \
	--data.train.dataset.class-files $list_dir/train_val_split/class_file.csv \
	--data.val.dataset.recordings-file $list_dir/wav.scp \
	--data.val.dataset.segments-file $list_dir/train_val_split/val_segments.csv \
	--in-model-file $nnet_s1 \
	--trainer.exp-path $nnet_s2_dir \
	--num-gpus $ngpu 
    
fi
