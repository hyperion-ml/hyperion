#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nnet_stage=1
config_file=default_config.sh
use_gpu=false
xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd --mem 4G"
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
elif [ $nnet_stage -eq 4 ];then
  nnet=$nnet_s4
  nnet_name=$nnet_s4_name
elif [ $nnet_stage -eq 5 ];then
  nnet=$nnet_s5
  nnet_name=$nnet_s5_name
elif [ $nnet_stage -eq 6 ];then
  nnet=$nnet_s6
  nnet_name=$nnet_s6_name
fi

xvector_dir=exp/xvectors/$nnet_name

if [[ $stage -le 1 && ( "$do_plda" == "true" || "$do_snorm" == "true" || "$do_qmf" == "true" || "$do_pca" == "true") ]]; then
  # Extract xvectors for training LDA/PLDA
  for name in voxceleb2cat_train
  do
    if [ $plda_num_augs -eq 0 ]; then
      steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd" --nj 100 ${xvec_args} \
	--random-utt-length true --min-utt-length 200 --max-utt-length 3000 \
	--feat-config $feat_config \
    	$nnet data/${name} \
    	$xvector_dir/${name}
    else
      steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd" --nj 300 ${xvec_args} \
	--random-utt-length true --min-utt-length 200 --max-utt-length 3000 \
	--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    	$nnet data/${name} \
    	$xvector_dir/${name}_augx${plda_num_augs} \
	data/${name}_augx${plda_num_augs}
    fi
  done
fi


if [ $stage -le 2 ]; then
  # Extracts x-vectors for evaluation
  if [ "$do_voxsrc22" == "true" ];then
    extra_data="voxsrc22_dev"
  fi
  for name in voxceleb1_test $extra_data
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 100 ? $num_spk:100))
    steps_xvec/extract_xvectors_from_wav.sh \
      --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
      --feat-config $feat_config \
      $nnet data/$name \
      $xvector_dir/$name
  done
fi

exit
