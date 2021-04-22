#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
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

attack_dir=exp/attacks/$nnet_name/
class_file=data/exp_attack_threat_model_v1/class2int
sign_nnet_dir=exp/sign_nnets/$nnet_name/exp_attack_threat_model_v1
sign_dir=exp/signatures/$nnet_name/exp_attack_threat_model_v1
logits_dir=exp/logits/$nnet_name/exp_attack_threat_model_v1
sign_nnet=$sign_nnet_dir/model_ep0020.pth
list_dir=data/exp_verif_attack_threat_model_v1
list_attack_type_dir=data/exp_verif_attack_type_v1

if [ $stage -le 1 ];then
    local/make_verif_test_lists_exp_attack_threat_model_v1.py \
	--input-file $attack_dir/pool_voxceleb1_test_v1/info.yml \
	--benign-wav-file data/voxceleb1_test/wav.scp \
	--output-dir $list_dir
fi


if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir \
	$sign_dir/voxceleb1_test
fi

proj_dir=$sign_dir/voxceleb1_test/tsne_attack_type
if [ $stage -le 3 ];then
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		steps_proj/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/voxceleb1_test/xvector.scp \
		--train-list $list_attack_type_dir/utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm --tsne-metric cosine --tsne-early-exaggeration $e --tsne-perplexity $p --tsne-init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

proj_dir=$sign_dir/voxceleb1_test/tsne_attack_threat_model
if [ $stage -le 4 ];then
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		steps_proj/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/voxceleb1_test/xvector.scp \
		--train-list $list_dir/utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm --tsne-metric cosine --tsne-early-exaggeration $e --tsne-perplexity $p --tsne-init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi


if [ $stage -le 5 ]; then
    # Eval attack logits
    mkdir -p $list_dir/voxceleb1_test
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir \
	$logits_dir/voxceleb1_test
fi

if [ $stage -le 6 ];then
    $train_cmd $logits_dir/voxceleb1_test/eval_acc.log \
        steps_proj/eval-classif-perf.py \
        --score-file scp:$logits_dir/voxceleb1_test/logits.scp \
        --key-file $list_dir/utt2attack \
	--class-file $class_file
fi


exit
