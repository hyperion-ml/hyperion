#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=1
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
    extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project voxceleb-dinossl.v1 --trainer.wandb.name $nnet_name.$(date -Iminutes)"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
    # dino arguments
    dinossl_args="--dinossl true "
    for arg in dinossl_nlayers dinossl_out_dim dinossl_use_bn_in_head dinossl_norm_last_layer \
        dinossl_local_crops_number dinossl_warmup_teacher_temp dinossl_teacher_temp \
        dinossl_warmup_teacher_temp_epochs dinossl_chunk_len_mult dinossl_reduce_overlap_prob; do
        if [ ! -z ${!arg} ]; then
            dinossl_args+="--${arg} ${!arg} " # ${!arg} return a value in the var, "${arg}"
        fi
    done
    echo "Dino arguments: ${dinossl_args}"

    # Edit train.scp and class2int files to ignore class balancing in batching (to
    # simulate a unsupervised scenario). Simply make class_idx == utt_idx
    # train.utt2utt.scp
    if [ ! -s ${list_dir}/lists_xvec/train.utt2utt.scp ]; then
        awk '{print $1" "$1}' ${list_dir}/lists_xvec/train.scp > ${list_dir}/lists_xvec/train.utt2utt.scp
    fi
    # (This block can be ignored) val.utt2utt.scp although it is not used in the end
    if [ ! -s ${list_dir}/lists_xvec/val.utt2utt.scp ]; then
        awk '{print $1" "$1}' ${list_dir}/lists_xvec/val.scp > ${list_dir}/lists_xvec/val.utt2utt.scp
    fi
    # utt2int
    if [ ! -s ${list_dir}/lists_xvec/utt2int ]; then
        cat <(awk '{print $1}' ${list_dir}/lists_xvec/train.scp) <(awk '{print $1}' ${list_dir}/lists_xvec/val.scp) > ${list_dir}/lists_xvec/utt2int
    fi
    


    mkdir -p $nnet_dir/log
    $cuda_cmd \
        --gpu $ngpu $nnet_dir/log/train.log \
        hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
        train_xvector_from_wav_dinossl.py $nnet_type --cfg $xvec_train_base_cfg $xvec_train_args $extra_args ${dinossl_args} \
        --data.train.dataset.audio-file $list_dir/wav.scp \
        --data.train.dataset.time-durs-file $list_dir/utt2dur \
        --data.train.dataset.key-file $list_dir/lists_xvec/train.utt2utt.scp \
        --data.train.dataset.class-file $list_dir/lists_xvec/utt2int \
        --data.val.dataset.audio-file $list_dir/wav.scp \
        --data.val.dataset.time-durs-file $list_dir/utt2dur \
        --data.val.dataset.key-file $list_dir/lists_xvec/val.utt2utt.scp \
        --trainer.exp-path $nnet_dir $args \
        --num-gpus $ngpu \
  
fi

