#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
use_gpu=false
#xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    eval_args="--use-gpu true"
    eval_cmd="$cuda_eval_cmd"
else
    eval_cmd="$train_cmd"
fi

output_dir=exp/recons_output/$nnet_name
if [[ "$model_type" =~ "vae" ]];then
    eval_script=hyp_utils/generative/eval_vae.sh
else
    echo "unknown model type $model_type"
    exit 1
fi

if [ $stage -le 1 ]; then
    for name in voxceleb1_test 
    do
	num_utt=$(wc -l data/$name/utt2spk | awk '{ print $1}')
	nj=$(($num_utt < 100 ? $num_utt:100))
	$eval_script --cmd "$eval_cmd --mem 6G" --nj $nj ${eval_args} \
	    $nnet data/$name $output_dir/$name
    done
fi




