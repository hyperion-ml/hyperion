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

sign_nnet_reldir=$spknet_name/$sign_nnet_name/$attack_type_split_tag
sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
sign_dir=exp/signatures/$sign_nnet_reldir
logits_dir=exp/logits/$sign_nnet_reldir
sign_nnet=$sign_nnet_dir/model_ep0020.pth
attack_dir=exp/attacks/$spknet_name
class_file=data/$attack_type_split_tag/class_file
list_dir=data/$spkverif_attack_type_split_tag

conda activate $HYP_ENV

if [ $stage -le 1 ];then
    echo "concatenate infos of all attacks types"
    mkdir -p $attack_dir/pool_voxceleb1_test_v1
    for attack in $known_attacks $unknown_attacks
    do
	for name in voxceleb1_test
	do
	    cat $attack_dir/$attack/$name/info/info.yaml
	done | awk '/attack_type:/ { sub(/pgd/,"'$attack'",$0);} 
                                   { print $0 }'
    done | awk '!/\{\}/' > $attack_dir/pool_voxceleb1_test_v1/info.yaml

    echo "make data directory for speaker verification attack classification by type"
    local/make_spkverif_test_lists_exp_attack_type_v1.py \
	--input-file $attack_dir/pool_voxceleb1_test_v1/info.yaml \
	--benign-wav-file data/voxceleb1_test/wav.scp \
	--output-dir $list_dir $spkverif_split_opts
fi

if [ $stage -le 2 ]; then
    echo "Extract attack signatures for spk. verif. attacks"
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir \
	$sign_dir/voxceleb1_test
fi

proj_dir=$sign_dir/voxceleb1_test/tsne
if [ $stage -le 3 ];then
    echo "Make TSNE plots on all test attacks"
    echo "Result will be left in $proj_dir"
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		steps_visual/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/voxceleb1_test/xvector.scp \
		--train-list $list_dir/utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm \
		--tsne.metric cosine --tsne.early-exaggeration $e --tsne.perplexity $p --tsne.init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

if [ $stage -le 4 ]; then
    echo "Eval signature network logits on test attacks"
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir \
	$logits_dir/voxceleb1_test
fi

if [ $stage -le 5 ];then
    echo "Compute cofusion matrices"
    echo "Result is left in $logits_dir/voxceleb1_test/eval_acc.log"
    $train_cmd $logits_dir/voxceleb1_test/eval_acc.log \
        steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/voxceleb1_test/logits.scp \
        --key-file $list_dir/utt2attack \
	--class-file $class_file
fi


exit
