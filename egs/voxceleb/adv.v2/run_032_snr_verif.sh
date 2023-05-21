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

#list with only the known attacks
list_someknown_dir=data/$sk_snr_split_tag
# list with all the attacks
list_all_dir=data/$snr_split_tag

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

sign_nnet_reldir=$spknet_name/$sign_nnet_name/$sk_snr_split_tag
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
	      train_xvector_from_wav.py  $sign_nnet_command --cfg $sign_nnet_config \
	      --data.train.dataset.audio-file $list_someknown_dir/trainval_wav.scp \
	      --data.train.dataset.time-durs-file $list_someknown_dir/trainval_utt2dur \
	      --data.train.dataset.segments-file $list_someknown_dir/train_utt2attack \
	      --data.train.dataset.class-file $list_someknown_dir/class_file \
	      --data.val.dataset.audio-file $list_someknown_dir/trainval_wav.scp \
	      --data.val.dataset.time-durs-file $list_someknown_dir/trainval_utt2dur \
	      --data.val.dataset.segments-file $list_someknown_dir/val_utt2attack \
	      --trainer.exp-path $sign_nnet_dir $args \
	      --num-gpus $ngpu 
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
        steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_all_dir/test_utt2attack \
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
    $list_all_dir/test_utt2attack \
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
	    data_dir=data/${snr_verif_split_tag}_enr${nes}sides
	    output_dir=$sign_dir/${snr_verif_split_tag}_enr${nes}sides_plda
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
	    data_dir=data/${snr_verif_split_tag}_enr${nes}sides
	    output_dir=$sign_dir/${snr_verif_split_tag}_enr${nes}sides_plda_bybook
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
if [ $stage -le 9 ]; then
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
if [ $stage -le 10 ]; then
    echo "Train PLDA"
    steps_be/train_be_v3.sh --cmd "$train_cmd" \
        --plda-type splda \
        --y-dim 6 \
	$sign_dir/train/xvector.scp \
        $list_dir/train \
        $be_dir &

    wait

fi

if [ $stage -le 11 ];then
    for nes in 1 3 5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes"
	(
	    data_dir=data/exp_attack_snr_verif_${nes}s_v2
	    output_dir=$sign_dir/attack_snr_verif_${nes}s_plda
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

if [ $stage -le 12 ];then
    for nes in  3 5 10 30 50 100
    do
	echo "Eval Attack Verification with PLDA scoring num sides=$nes"
	(
	    data_dir=data/exp_attack_snr_verif_${nes}s_v2
	    output_dir=$sign_dir/attack_snr_verif_${nes}s_plda_bybook
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
