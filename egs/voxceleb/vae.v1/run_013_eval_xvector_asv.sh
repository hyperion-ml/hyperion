#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
use_gpu=false
xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

xvector_dir=exp/xvectors/$nnet_name/$xvec_nnet_name
score_be_dir=exp/scores/$nnet_name/$xvec_nnet_name/cosine


if [ $stage -le 1 ]; then
    # Extracts x-vectors for evaluation
    for name in voxceleb1_test 
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors_with_vae_preproc.sh \
	    --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    $xvec_nnet $nnet data/$name \
	    $xvector_dir/$name
    done
fi


if [ $stage -le 2 ];then

    echo "Eval Voxceleb 1 with Cosine scoring"
    steps_be/eval_be_cos.sh --cmd "$train_cmd" \
    	data/voxceleb1_test/trials \
    	data/voxceleb1_test/utt2model \
    	$xvector_dir/voxceleb1_test/xvector.scp \
    	$score_be_dir/voxceleb1_scores

    $train_cmd --mem 10G --num-threads 6 $score_be_dir/log/score_voxceleb1.log \
	local/score_voxceleb1.sh data/voxceleb1_test $score_be_dir 

    for f in $(ls $score_be_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi

exit
