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
resume=false
interactive=false
num_workers=8
xvec_use_gpu=false
xvec_chunk_length=12800

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ "$xvec_use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

batch_size=$(($batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $eff_batch_size | awk '{ print int($2/$1+0.5)}')
log_interval=$(echo 100*$grad_acc_steps | bc)
list_dir=data/exp_attack_type_v2
list_test_dir=data/exp_attack_type_v1

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

sign_nnet_dir=exp/sign_nnets/$nnet_name/exp_attack_type_v2
sign_dir=exp/signatures/$nnet_name/exp_attack_type_v2
logits_dir=exp/logits/$nnet_name/exp_attack_type_v2
nnet_num_epochs=20
sign_nnet=$sign_nnet_dir/model_ep0020.pth
margin=0.2
margin_warmup=6
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yml"
aug_opt=""
embed_dim=10
lr=0.01

opt_opt="--opt-optimizer adam --opt-lr $lr --opt-beta1 0.9 --opt-beta2 0.95 --opt-weight-decay 1e-5 --opt-amsgrad --use-amp"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 2000 --lrsch-hold-steps 4000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 16000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"

# Network Training
if [ $stage -le 1 ]; then

    if [[ ${nnet_type} =~ resnet ]] || [[ ${nnet_type} =~ resnext ]]; then
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

    mkdir -p $sign_nnet_dir/log
    $cuda_cmd --gpu $ngpu $sign_nnet_dir/log/train.log \
	hyp_utils/torch.sh --num-gpus $ngpu \
	$train_exec  @$feat_config $aug_opt \
	--audio-path $list_dir/trainval_wav.scp \
	--time-durs-file $list_dir/trainval_utt2dur \
	--train-list $list_dir/train_utt2attack \
	--val-list $list_dir/val_utt2attack \
	--class-file $list_dir/class2int \
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
	--exp-path $sign_nnet_dir $args

fi

if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    mkdir -p $list_test_dir/test
    cp $list_test_dir/test_wav.scp $list_test_dir/test/wav.scp
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_test_dir/test \
	$sign_dir/test
fi

proj_dir=$sign_dir/test/tsne
if [ $stage -le 3 ];then
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		steps_proj/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_test_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm --tsne-metric cosine --tsne-early-exaggeration $e --tsne-perplexity $p --tsne-init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

if [ $stage -le 4 ]; then
    # Eval attack logits
    mkdir -p $list_test_dir/test
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_test_dir/test \
	$logits_dir/test
fi

if [ $stage -le 5 ];then
    $train_cmd $logits_dir/eval_acc.log \
        steps_proj/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_test_dir/test_utt2attack \
	--class-file $list_dir/class2int         
fi


exit
