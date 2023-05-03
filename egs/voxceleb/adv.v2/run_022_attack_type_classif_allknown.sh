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

list_dir=data/$attack_type_split_tag

args=""

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

sign_nnet_reldir=$spknet_name/$sign_nnet_name/$attack_type_split_tag
sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
sign_dir=exp/signatures/$sign_nnet_reldir
logits_dir=exp/logits/$sign_nnet_reldir
sign_nnet=$sign_nnet_dir/model_ep0020.pth

# Network Training
if [ $stage -le 1 ]; then
    echo "Train signature network on all attacks"
    mkdir -p $sign_nnet_dir/log
    $cuda_cmd --gpu $ngpu $sign_nnet_dir/log/train.log \
	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	train_xvector_from_wav.py  $sign_nnet_command --cfg $sign_nnet_config \
	--data.train.dataset.audio-file $list_dir/trainval_wav.scp \
	--data.train.dataset.time-durs-file $list_dir/trainval_utt2dur \
	--data.train.dataset.segments-file $list_dir/train_utt2attack \
	--data.train.dataset.class-file $list_dir/class_file \
	--data.val.dataset.audio-file $list_dir/trainval_wav.scp \
	--data.val.dataset.time-durs-file $list_dir/trainval_utt2dur \
	--data.val.dataset.segments-file $list_dir/val_utt2attack \
	--trainer.exp-path $sign_nnet_dir $args \
	--num-gpus $ngpu \

fi

if [ $stage -le 2 ]; then
    echo "Extract signatures on the test set"
    mkdir -p $list_dir/test
    cp $list_dir/test_wav.scp $list_dir/test/wav.scp
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir/test \
	$sign_dir/test
fi

proj_dir=$sign_dir/test/tsne_${attack_type_split_tag}
if [ $stage -le 3 ];then
    echo "Make TSNE plots on all test attacks"
    echo "Result will be left in $proj_dir"
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		hyp_utils/conda_env.sh steps_visual/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm --tsne.metric cosine --tsne.early-exaggeration $e --tsne.perplexity $p --tsne.init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

if [ $stage -le 4 ]; then
    echo "Eval signature network logits on test attacks"
    mkdir -p $list_dir/test
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir/test \
	$logits_dir/test
fi

if [ $stage -le 5 ];then
    echo "Compute confusion matrices"
    echo "Result is left in $logits_dir/test/eval_acc.log"
    $train_cmd $logits_dir/test/eval_acc.log \
        hyp_utils/conda_env.sh steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_dir/test_utt2attack \
	--class-file $list_dir/class_file
fi


exit
