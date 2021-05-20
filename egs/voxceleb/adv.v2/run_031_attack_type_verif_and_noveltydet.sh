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
list_someknown_dir=data/$sk_attack_type_split_tag
# list with all the attacks
list_all_dir=data/$attack_type_split_tag

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

sign_nnet_reldir=$spknet_name/$sign_nnet_name/$sk_attack_type_split_tag
sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
sign_dir=exp/signatures/$sign_nnet_reldir
logits_dir=exp/logits/$sign_nnet_reldir
sign_nnet=$sign_nnet_dir/model_ep0020.pth

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
    mkdir -p $list_all_dir/test
    cp $list_all_dir/test_wav.scp $list_all_dir/test/wav.scp
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_all_dir/test \
	$sign_dir/test
fi

proj_dir=$sign_dir/test/tsne_all
if [ $stage -le 3 ];then
    echo "Plot TSNE for known and unknown attacks"
    echo "Result will be in $proj_dir"
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		hyp_utils/conda_env.sh steps_visual/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_all_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm \
		--tsne.metric cosine --tsne.early-exaggeration $e --tsne.perplexity $p --tsne.init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

if [ $stage -le 4 ]; then
    echo "Compute logits for all attacks"
    mkdir -p $list_all_dir/test
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_all_dir/test \
	$logits_dir/test
fi

if [ $stage -le 5 ];then
    echo "Compute confusion matrices from logits using all attacks"
    echo "Result left in $logits_dir/test_all/eval_acc.log"
    $train_cmd $logits_dir/test_all/eval_acc.log \
	hyp_utils/conda_env.sh steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_all_dir/test_utt2attack \
	--class-file $list_someknown_dir/class_file

fi

# if [ $stage -le 6 ];then
#     for nes in 1 #3 5 10
#     do
# 	echo "Eval Attack Verification with Cosine scoring num sides=$nes"
# 	(
# 	    data_dir=data/exp_attack_type_verif_${nes}s_v2
# 	    output_dir=$sign_dir/attack_type_verif_${nes}s
# 	    steps_backend/eval_be_cos_Nvs1.sh --cmd "$train_cmd" --num-parts 2 \
# 		$data_dir/trials \
# 		$data_dir/utt2enr \
# 		$sign_dir/test/xvector.scp \
# 		$output_dir/attack_verif_scores
	    
# 	    $train_cmd --mem 10G $output_dir/log/score_attack_verif.log \
# 		steps_backend/score_attack_verif.sh $data_dir $output_dir
	    
# 	    for f in $(ls $output_dir/*_results);
# 	    do
# 		echo $f
# 		cat $f
# 		echo ""
# 	    done
# 	) &
#     done
#     wait
# fi

if [ $stage -le 7 ]; then
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
if [ $stage -le 8 ]; then
    echo "Train PLDA model on known attacks"
    steps_backend/train_be_v1.sh --cmd "$train_cmd" \
        --plda-type splda \
        --y-dim 6 \
	$sign_dir/train/xvector.scp \
        $list_someknown_dir/train \
        $be_dir
fi

if [ $stage -le 9 ];then
    for nes in  1 3 #5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes with known and unknown attacks"
	(
	    data_dir=data/${attack_type_verif_split_tag}_enr${nes}sides
	    output_dir=$sign_dir/${attack_type_verif_split_tag}_enr${nes}sides_plda
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

if [ $stage -le 10 ];then
    for nes in  3 #5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes bythebook scoring"
	(
	    data_dir=data/${attack_type_verif_split_tag}_enr${nes}sides
	    output_dir=$sign_dir/${attack_type_verif_split_tag}_enr${nes}sides_plda_bybook
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

if [ $stage -le 11 ];then
    echo "Eval Attack Novelty with PLDA"
    data_dir=data/${novelty_split_tag}
    output_dir=$sign_dir/${novelty_split_tag}_plda
    echo "Results with be in $output_dir"
    mkdir -p $sign_dir/train_test
    cat $sign_dir/train/xvector.scp $sign_dir/test/xvector.scp > $sign_dir/train_test/xvector.scp
    steps_backend/eval_be_novelty.sh --cmd "$train_cmd" \
	--plda-type splda \
	$data_dir/trials \
	$list_someknown_dir/train/utt2spk \
	$sign_dir/train_test/xvector.scp \
	$be_dir/lnorm.h5 \
	$be_dir/plda.h5 \
	$output_dir/attack_novelty_scores
    
    $train_cmd --mem 10G $output_dir/log/score_attack_novelty.log \
	hyp_utils/conda_env.sh local/score_dcf.py --key-file $data_dir/trials \
	--score-file $output_dir/attack_novelty_scores --output-path $output_dir/attack_novelty
    
    for f in $(ls $output_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done
fi

if [ $stage -le 12 ];then
    echo "Eval Attack Novelty with PLDA using by-the-book evaluation"
    data_dir=data/${novelty_split_tag}
    output_dir=$sign_dir/${novelty_split_tag}_plda_bybook
    echo "Results with be in $output_dir"
    steps_backend/eval_be_novelty.sh --cmd "$train_cmd" \
	--plda-type splda --plda-opts "--eval-method book" \
	$data_dir/trials \
	$list_someknown_dir/train/utt2spk \
	$sign_dir/train_test/xvector.scp \
	$be_dir/lnorm.h5 \
	$be_dir/plda.h5 \
	$output_dir/attack_novelty_scores
    
    $train_cmd --mem 10G $output_dir/log/score_attack_novelty.log \
	hyp_utils/conda_env.sh local/score_dcf.py \
	--key-file $data_dir/trials --score-file $output_dir/attack_novelty_scores \
	--output-path $output_dir/attack_novelty
    
    for f in $(ls $output_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done
fi


be_dir=$sign_dir/train_nobenign
if [ $stage -le 13 ]; then
    echo "Train PLDA on known attacks without benign samples"
    mkdir -p $list_someknown_dir/train_nobenign
    awk '!/benign/' $list_someknown_dir/train/utt2spk > $list_someknown_dir/train_nobenign/utt2spk
    steps_backend/train_be_v1.sh --cmd "$train_cmd" \
        --plda-type splda \
        --y-dim 6 \
	$sign_dir/train/xvector.scp \
        $list_someknown_dir/train_nobenign \
        $be_dir 
    
fi

if [ $stage -le 14 ];then
    for nes in 1 3 #5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes without benign class in test"
	(
	    data_dir0=data/${attack_type_verif_split_tag}_enr${nes}sides
	    data_dir=data/${attack_type_verif_split_tag}_enr${nes}sides_nobenign
	    output_dir=$sign_dir/${attack_type_verif_split_tag}_enr${nes}sides_plda_nobenign
	    echo "Results with be in $output_dir"
	    mkdir -p $data_dir
	    for f in trials trials_known trials_unknown utt2enr
	    do
		if [ $nes -eq 1 ];then
		    awk '($1 ~ /fgsm/ || $1 ~ /pgd/ || $1 ~ /cw/) && ($2 ~ /fgsm/ || $2 ~ /pgd/ || $2 ~ /cw/)' $data_dir0/$f > $data_dir/$f
		else
		    awk '!/benign/ && ($2 ~ /fgsm/ || $2 ~ /pgd/ || $2 ~ /cw/)' $data_dir0/$f > $data_dir/$f
		fi
	    done
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

if [ $stage -le 15 ];then
    for nes in 3 #5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes by-the-book scoring without benign class in test"
	(
	    data_dir=data/${attack_type_verif_split_tag}_enr${nes}sides_nobenign
	    output_dir=$sign_dir/${attack_type_verif_split_tag}_enr${nes}sides_plda_bybook_nobenign
	    echo "Results with be in $output_dir"
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

if [ $stage -le 16 ];then
    echo "Eval Attack Novelty with PLDA without benign class in test"
    data_dir=data/${novelty_split_tag}
    output_dir=$sign_dir/${novelty_split_tag}_plda_nobenign
    echo "Results with be in $output_dir"
    steps_backend/eval_be_novelty.sh --cmd "$train_cmd" \
	--plda-type splda \
	$data_dir/trials_nobenign \
	$list_someknown_dir/train_nobenign/utt2spk \
	$sign_dir/train_test/xvector.scp \
	$be_dir/lnorm.h5 \
	$be_dir/plda.h5 \
	$output_dir/attack_novelty_scores
    
    $train_cmd --mem 10G $output_dir/log/score_attack_novelty.log \
	hyp_utils/conda_env.sh local/score_dcf.py --key-file $data_dir/trials_nobenign \
	--score-file $output_dir/attack_novelty_scores --output-path $output_dir/attack_novelty
    
    for f in $(ls $output_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done
fi

if [ $stage -le 17 ];then
    echo "Eval Attack Novelty with PLDA bythebook scoring without benign class in test"
    data_dir=data/${novelty_split_tag}
    output_dir=$sign_dir/${novelty_split_tag}_plda_bybook_nobenign
    echo "Results with be in $output_dir"
    steps_backend/eval_be_novelty.sh --cmd "$train_cmd" \
	--plda-type splda --plda-opts "--eval-method book" \
	$data_dir/trials_nobenign \
	$list_someknown_dir/train_nobenign/utt2spk \
	$sign_dir/train_test/xvector.scp \
	$be_dir/lnorm.h5 \
	$be_dir/plda.h5 \
	$output_dir/attack_novelty_scores
    
    $train_cmd --mem 10G $output_dir/log/score_attack_novelty.log \
	hyp_utils/conda_env.sh local/score_dcf.py --key-file $data_dir/trials_nobenign \
	--score-file $output_dir/attack_novelty_scores --output-path $output_dir/attack_novelty
    
    for f in $(ls $output_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done
fi
