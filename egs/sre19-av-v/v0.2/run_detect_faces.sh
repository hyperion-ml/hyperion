#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
# This scripts only runs the face detector for visual inspection
# It is not needed in the pipeline because the embedding extraction
# script, runs the face detector on-the-fly
#
. ./cmd.sh
. ./path.sh
set -e
nodes=fs01
storage_name=$(date +'%m_%d_%H_%M')
face_det_dir=`pwd`/exp/face_det
use_gpu=false
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
  cmd="$cuda_eval_cmd"
else
  cmd="$train_cmd"
fi

if [ $stage -le 1 ];then 
  for name in sre19_av_v_dev_test \
		sre19_av_v_eval_test \
		janus_dev_enroll \
		janus_dev_test_core \
		janus_eval_enroll \
		janus_eval_test_core 
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 40 ? $num_spk:40))
    steps_insightface/eval_face_det_v1.sh \
      --cmd "$cmd" --nj $nj --use-gpu $use_gpu \
      data/$name $face_det_model $face_det_dir/$name
  done
fi


