#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2
config_file=default_config.sh
use_gpu=false
nnet_stage=3
hf_chunk_length=120 #seconds
xvec_chunk_length=120 #seconds
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu true --xvec-chunk-length $xvec_chunk_length --hf-chunk-length $hf_chunk_length"
  xvec_cmd="$cuda_eval_cmd --mem 6G"
else
  xvec_cmd="$train_cmd --mem 12G"
fi

if [ $nnet_stage -eq 1 ];then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ];then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
elif [ $nnet_stage -eq 3 ];then
  nnet=$nnet_s3
  nnet_name=$nnet_s3_name
fi

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ]; then
  # Extract xvectors for training LDA/PLDA
  for name in voxceleb2cat_train
  do
    if [ $plda_num_augs -eq 0 ]; then
      steps_xvec/extract_wav2vec2xvectors.sh \
	--cmd "$xvec_cmd" --nj 100 ${xvec_args} \
	--random-utt-length true --min-utt-length 4 --max-utt-length 140 \
    	$nnet data/${name} \
    	$xvector_dir/${name}
    else
      steps_xvec/extract_wav2vec2xvectors.sh \
	--cmd "$xvec_cmd" --nj 300 ${xvec_args} \
	--random-utt-length true --min-utt-length 4 --max-utt-length 140 \
	--aug-config $plda_aug_config --num-augs $plda_num_augs \
    	$nnet data/${name} \
    	$xvector_dir/${name}_augx${plda_num_augs} \
	data/${name}_augx${plda_num_augs}
    fi
  done
fi

if [ $stage -le 2 ]; then
  # Extracts x-vectors for evaluation
  for name in voxceleb1_test 
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 100 ? $num_spk:100))
    steps_xvec/extract_wav2vec2xvectors.sh \
      --cmd "$xvec_cmd" --nj $nj ${xvec_args} \
      $nnet data/$name \
      $xvector_dir/$name
  done
fi

exit
