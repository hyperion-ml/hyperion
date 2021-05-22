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
num_workers=4
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

batch_size=$(($sign_nnet_batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $sign_nnet_eff_batch_size | awk '{ print int($2/$1+0.5)}')
log_interval=$(echo 100*$grad_acc_steps | bc)
#list with only the known attacks
list_someknown_dir=data/$sk_threat_model_split_tag
# list with all the attacks
list_all_threat_model_dir=data/$threat_model_split_tag
list_all_attack_type_dir=data/$attack_type_split_tag

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

sign_nnet_reldir=$spknet_name/$sign_nnet_name/$sk_threat_model_split_tag
sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
sign_dir=exp/signatures/$sign_nnet_reldir
logits_dir=exp/logits/$sign_nnet_reldir
sign_nnet=$sign_nnet_dir/model_ep0015.pth

# Network Training
if [ $stage -le 1 ]; then
    echo "Train attack signature network on known attacks only"
    mkdir -p $sign_nnet_dir/log
    $cuda_cmd --gpu $ngpu $sign_nnet_dir/log/train.log \
	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	torch-train-xvec-from-wav.py  $sign_nnet_command --cfg $sign_nnet_config \
	--audio-path $list_someknown_dir/trainval_wav.scp \
	--time-durs-file $list_someknown_dir/trainval_utt2dur \
	--train-list $list_someknown_dir/train_utt2attack \
	--val-list $list_someknown_dir/val_utt2attack \
	--class-file $list_someknown_dir/class_file \
	--batch-size $batch_size \
	--num-workers $num_workers \
	--grad-acc-steps $grad_acc_steps \
	--num-gpus $ngpu \
	--log-interval $log_interval \
	--exp-path $sign_nnet_dir $args
fi

if [ $stage -le 2 ]; then
    echo "Extract attack signatures for known and unknown attacks"
    mkdir -p $list_all_attack_type_dir/test
    cp $list_all_attack_type_dir/test_wav.scp $list_all_attack_type_dir/test/wav.scp
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_all_attack_type_dir/test \
	$sign_dir/test
fi

proj_dir=$sign_dir/test/tsne_attack_type
if [ $stage -le 3 ];then
    echo "Plot TSNE for known and unknown attacks where colors are attack types"
    echo "Result will be in $proj_dir"
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		hyp_utils/conda_env.sh steps_visual/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_all_attack_type_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm \
		--tsne.metric cosine --tsne.early-exaggeration $e --tsne.perplexity $p --tsne.init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

proj_dir=$sign_dir/test/tsne_threat_model
if [ $stage -le 4 ];then
    echo "Plot TSNE for known and unknown attacks where colors are threat models"
    echo "Result will be in $proj_dir"
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		hyp_utils/conda_env.sh steps_visual/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_all_threat_model_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm \
		--tsne.metric cosine --tsne.early-exaggeration $e --tsne.perplexity $p --tsne.init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

if [ $stage -le 4 ]; then
    # Eval attack logits
    echo "Compute logits for all attacks"
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_all_threat_model_dir/test \
	$logits_dir/test
fi

if [ $stage -le 5 ];then
    echo "Compute confusion matrices from logits using all attacks"
    echo "Result left in $logits_dir/test_all/eval_acc.log"
    $train_cmd $logits_dir/test_all/eval_acc.log \
        steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_all_threat_model_dir/test_utt2attack \
	--class-file $list_someknown_dir/class_file         
fi

if [ $stage -le 6 ];then
    echo "Compute confusion matrices from logits using only known attacks"
    echo "Result left in $logits_dir/test_known/eval_acc.log"
    $train_cmd $logits_dir/test_known/eval_acc.log \
        steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_someknown_dir/test_utt2attack \
	--class-file $list_someknown_dir/class_file
fi


if [ $stage -le 7 ];then
    echo "Compute confusion matrices from logits using only unknown attacks"
    echo "Result left in $logits_dir/test_unknown/eval_acc.log"

    mkdir -p $logits_dir/test_unknown
    awk -v f=$list_someknown_dir/test_utt2attack 'BEGIN{
while(getline < f)
{
  v[$1]=1
}
}
!/benign/{ if(!($1 in v)){ print $0}}' \
    $list_all_threat_model_dir/test_utt2attack \
    > $logits_dir/test_unknown/utt2attack
    
    $train_cmd $logits_dir/test_unknown/eval_acc.log \
        steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $logits_dir/test_unknown/utt2attack \
	--class-file $list_someknown_dir/class_file
fi

if [ $stage -le 9 ]; then
    echo "Extracts x-vectors to train plda on known attacks"
    mkdir -p $list_someknown_dir/train
    cp $list_someknown_dir/train_wav.scp $list_someknown_dir/train/wav.scp
    cp $list_someknown_dir/train_utt2attack $list_someknown_dir/train/utt2spk
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_someknown_dir/train \
	$sign_dir/train
fi

be_dir=$sign_dir/train
if [ $stage -le 10 ]; then
    echo "Train PLDA model on known attacks"
    steps_backend/train_be_v1.sh --cmd "$train_cmd" \
        --plda-type splda \
        --y-dim 6 \
	$sign_dir/train/xvector.scp \
        $list_someknown_dir/train \
        $be_dir
fi

if [ $stage -le 11 ];then
    for nes in  1 3 #5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes with known and unknown attacks"
	(
	    data_dir=data/${threat_model_verif_split_tag}_enr${nes}sides
	    output_dir=$sign_dir/${threat_model_verif_split_tag}_enr${nes}sides_plda
	    echo "Results will be in $output_dir"
	    steps_backend/eval_be_Nvs1_v1.sh --cmd "$train_cmd" --num-parts 2 \
		--plda-type splda \
		$data_dir/trials \
		$data_dir/utt2enr \
		$sign_dir/test/xvector.scp \
		$be_dir/lnorm.h5 \
		$be_dir/plda.h5 \
		$output_dir/attack_verif_scores
	    
	    $train_cmd --mem 10G $output_dir/log/score_attack_verif.log \
		steps_backend/score_attack_verif.sh $data_dir $output_dir
	    
	    for f in $(ls $output_dir/*_results);
	    do
		echo $f
		cat $f
		echo ""
	    done
	) &
    done
    wait
fi

if [ $stage -le 12 ];then
    for nes in  3 #5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes bythebook scoring"
	(
	    data_dir=data/${threat_model_verif_split_tag}_enr${nes}sides
	    output_dir=$sign_dir/${threat_model_verif_split_tag}_enr${nes}sides_plda_bybook
	    echo "Results will be in $output_dir"
	    steps_backend/eval_be_Nvs1_v1.sh --cmd "$train_cmd" --num-parts 2 \
		--plda-type splda --plda-opts "--eval-method book" \
		$data_dir/trials \
		$data_dir/utt2enr \
		$sign_dir/test/xvector.scp \
		$be_dir/lnorm.h5 \
		$be_dir/plda.h5 \
		$output_dir/attack_verif_scores
	    
	    $train_cmd --mem 10G $output_dir/log/score_attack_verif.log \
		steps_backend/score_attack_verif.sh $data_dir $output_dir
	    
	    for f in $(ls $output_dir/*_results);
	    do
		echo $f
		cat $f
		echo ""
	    done
	) &
    done
    wait
fi



exit



exit
#########################################
proj_dir=$sign_dir/test/tsne_attack_type
if [ $stage -le 3 ];then
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		steps_proj/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_test_attack_type_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm --tsne-metric cosine --tsne-early-exaggeration $e --tsne-perplexity $p --tsne-init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

proj_dir=$sign_dir/test/tsne_threat_model
if [ $stage -le 4 ];then
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		steps_proj/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_test_threat_model_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm --tsne-metric cosine --tsne-early-exaggeration $e --tsne-perplexity $p --tsne-init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi


if [ $stage -le 5 ]; then
    # Eval attack logits
    #mkdir -p $list_test_threat_model_dir/test
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_test_threat_model_dir/test \
	$logits_dir/test
fi

if [ $stage -le 6 ];then
    $train_cmd $logits_dir/eval_acc.log \
        steps_proj/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_test_threat_model_dir/test_utt2attack \
	--class-file $list_dir/class2int         
fi


if [ $stage -le 7 ];then
    $train_cmd $logits_dir/test_seen/eval_acc.log \
        steps_proj/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_dir/test_utt2attack \
	--class-file $list_dir/class2int         
fi

if [ $stage -le 8 ];then
    mkdir -p $logits_dir/test_unseen
    awk -v f=$list_dir/test_utt2attack 'BEGIN{
while(getline < f)
{
  v[$1]=1
}
}
!/benign/{ if(!($1 in v)){ print $0}}' \
    $list_test_threat_model_dir/test_utt2attack \
    > $logits_dir/test_unseen/utt2attack
    
    $train_cmd $logits_dir/test_unseen/eval_acc.log \
        steps_proj/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $logits_dir/test_unseen/utt2attack \
	--class-file $list_dir/class2int         
fi

# if [ $stage -le 9 ];then
#     echo "Eval Attack Verification with Cosine scoring"
#     steps_be/eval_be_cos.sh --cmd "$train_cmd" --num-parts 20 \
#         data/exp_attack_threat_model_verif_v2/trials \
#         data/exp_attack_threat_model_verif_v2/utt2enr \
#         $sign_dir/test/xvector.scp \
#         $sign_dir/test_scores/attack_verif_scores

#     $train_cmd --mem 10G $sign_dir/test_scores/log/score_attack_verif.log \
#         steps_proj/score_attack_verif.sh data/exp_attack_threat_model_verif_v2 $sign_dir/test_scores

#     for f in $(ls $sign_dir/test_scores/*_results);
#     do
#         echo $f
#         cat $f
#         echo ""
#     done
# fi

if [ $stage -le 9 ];then
    for nes in 1 3 5 10
    do
	echo "Eval Attack Verification with Cosine scoring num sides=$nes"
	(
	    data_dir=data/exp_attack_threat_model_verif_${nes}s_v2
	    output_dir=$sign_dir/attack_threat_model_verif_${nes}s
	    steps_be/eval_be_cos_Nvs1.sh --cmd "$train_cmd" --num-parts 2 \
		$data_dir/trials \
		$data_dir/utt2enr \
		$sign_dir/test/xvector.scp \
		$output_dir/attack_verif_scores
	    
	    $train_cmd --mem 10G $output_dir/log/score_attack_verif.log \
		steps_proj/score_attack_verif.sh $data_dir $output_dir
	    
	    for f in $(ls $output_dir/*_results);
	    do
		echo $f
		cat $f
		echo ""
	    done
	) &
    done
    wait
fi

if [ $stage -le 10 ]; then
    # Extracts x-vectors for training
    mkdir -p $list_dir/train
    cp $list_dir/train_wav.scp $list_dir/train/wav.scp
    cp $list_dir/train_utt2attack $list_dir/train/utt2spk
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir/train \
	$sign_dir/train
fi

be_dir=$sign_dir/train
if [ $stage -le 11 ]; then
    echo "Train PLDA"
    steps_be/train_be_v3.sh --cmd "$train_cmd" \
        --plda-type splda \
        --y-dim 6 \
	$sign_dir/train/xvector.scp \
        $list_dir/train \
        $be_dir &

    wait

fi

if [ $stage -le 12 ];then
    for nes in 1 3 5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes"
	(
	    data_dir=data/exp_attack_threat_model_verif_${nes}s_v2
	    output_dir=$sign_dir/attack_threat_model_verif_${nes}s_plda
	    steps_be/eval_be_Nvs1_v1.sh --cmd "$train_cmd" --num-parts 2 \
		--plda-type splda \
		$data_dir/trials \
		$data_dir/utt2enr \
		$sign_dir/test/xvector.scp \
		$be_dir/lnorm.h5 \
		$be_dir/plda.h5 \
		$output_dir/attack_verif_scores
	    
	    $train_cmd --mem 10G $output_dir/log/score_attack_verif.log \
		steps_proj/score_attack_verif.sh $data_dir $output_dir
	    
	    for f in $(ls $output_dir/*_results);
	    do
		echo $f
		cat $f
		echo ""
	    done
	) &
    done
    wait
fi

if [ $stage -le 13 ];then
    for nes in  3 5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes"
	(
	    data_dir=data/exp_attack_threat_model_verif_${nes}s_v2
	    output_dir=$sign_dir/attack_threat_model_verif_${nes}s_plda_bybook
	    steps_be/eval_be_Nvs1_v1.sh --cmd "$train_cmd" --num-parts 2 \
		--plda-type splda --plda-opts "--eval-method book" \
		$data_dir/trials \
		$data_dir/utt2enr \
		$sign_dir/test/xvector.scp \
		$be_dir/lnorm.h5 \
		$be_dir/plda.h5 \
		$output_dir/attack_verif_scores
	    
	    $train_cmd --mem 10G $output_dir/log/score_attack_verif.log \
		steps_proj/score_attack_verif.sh $data_dir $output_dir
	    
	    for f in $(ls $output_dir/*_results);
	    do
		echo $f
		cat $f
		echo ""
	    done
	) &
    done
    wait
fi

exit

