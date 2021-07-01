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
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

nnet_dir=${nnet_dir}_Noclassbalancebatching_correct
batch_size=$(($batch_size_1gpu*$ngpu))
#grad_acc_steps=$(echo $batch_size $eff_batch_size | awk '{ print int($2/$1+0.5)}')
grad_acc_steps=1 # JJ: TODO - Do not apply grad_acc_steps with the current dinossl (i.e., set this to 1). Instead, apply linear scaling of the lr as is done in original dino repo.
log_interval=$(echo 100*$grad_acc_steps | bc)
list_dir=data/${nnet_data}_proc_audio_no_sil

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi
if [ "$use_tb" == "true" ];then
    args="$args --use-tensorboard"
fi
if [ "$use_wandb" == "true" ];then
    args="$args --use-wandb --wandb.project voxceleb-v1.1 --wandb.name $nnet_name.$(date -Iminutes)"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then

    if [[ ${nnet_type} =~ resnet ]] || [[ ${nnet_type} =~ resnext ]] || [[ ${nnet_type} =~ res2net ]] || [[ ${nnet_type} =~ res2next ]]; then
	train_exec=torch-train-resnet-xvec-from-wav.py
    elif [[ ${nnet_type} =~ efficientnet ]]; then
	train_exec=torch-train-efficientnet-xvec-from-wav.py
    elif [[ ${nnet_type} =~ tdnn ]]; then
	train_exec=torch-train-tdnn-xvec-from-wav.py
    elif [[ ${nnet_type} =~ transformer ]]; then
	train_exec=torch-train-transformer-xvec-v1-from-wav.py
    else
	echo "$nnet_type not supported"
	exit 1
    fi

    # add dinossl-related parameters
    if [[ ${dinossl} == true ]]; then
        # dino arguments
        train_exec=${train_exec%.py}_dinossl.py
        dinossl_args="--dinossl true "
        for arg in dinossl_nlayers dinossl_out_dim dinossl_use_bn_in_head dinossl_norm_last_layer \
            dinossl_local_crops_number dinossl_warmup_teacher_temp dinossl_teacher_temp \
            dinossl_warmup_teacher_temp_epochs dinossl_chunk_len_mult; do
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
        # val.utt2utt.scp although it is not used in the end
        if [ ! -s ${list_dir}/lists_xvec/val.utt2utt.scp ]; then
            awk '{print $1" "$1}' ${list_dir}/lists_xvec/val.scp > ${list_dir}/lists_xvec/val.utt2utt.scp
        fi
        # utt2int
        if [ ! -s ${list_dir}/lists_xvec/utt2int ]; then
            cat <(awk '{print $1}' ${list_dir}/lists_xvec/train.scp) <(awk '{print $1}' ${list_dir}/lists_xvec/val.scp) > ${list_dir}/lists_xvec/utt2int
        fi
    fi

    mkdir -p $nnet_dir/log
    $cuda_cmd --gpu $ngpu $nnet_dir/log/train.log \
	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	$train_exec --feats $feat_config $aug_opt \
	--audio-path $list_dir/wav.scp \
	--time-durs-file $list_dir/utt2dur \
	--train-list $list_dir/lists_xvec/train.utt2utt.scp \
	--val-list $list_dir/lists_xvec/val.utt2utt.scp \
	--class-file $list_dir/lists_xvec/utt2int \
	--min-chunk-length $min_chunk --max-chunk-length $max_chunk \
	--iters-per-epoch $ipe \
	--batch-size $batch_size \
	--num-workers $num_workers \
	--grad-acc-steps $grad_acc_steps \
	--embed-dim $embed_dim $nnet_opt $opt_opt $lrs_opt \
	--epochs $nnet_num_epochs \
	--s $s --margin $margin --margin-warmup-epochs $margin_warmup \
	--dropout-rate $dropout \
	--num-gpus $ngpu \
	--log-interval $log_interval \
	--exp-path $nnet_dir $args \
    ${dinossl_args}

fi


exit
