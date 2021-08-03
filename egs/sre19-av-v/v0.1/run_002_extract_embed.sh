#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e
nodes=fs01
storage_name=$(date +'%m_%d_%H_%M')
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

face_embed_ref_dir=$face_embed_dir/ref
face_embed_facedet_dir=$face_embed_dir/facedet

if [ $stage -le 1 ];then 
  for name in janus_dev_enroll janus_eval_enroll 
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 40 ? $num_spk:40))
    steps_insightface/extract_face_embed_from_bbox_plus_face_det_v4.sh \
      --cmd "$cmd" --nj $nj --use-gpu $use_gpu \
      --det-window 61 \
      data/$name $face_det_model $face_reco_model $face_embed_ref_dir/$name
  done
fi


if [ $stage -le 2 ];then 
  for name in sre19_av_v_dev_enroll sre19_av_v_eval_enroll 
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 40 ? $num_spk:40))
    steps_insightface/extract_face_embed_from_bbox_plus_face_det_v4.sh \
      --cmd "$cmd" --nj $nj --use-gpu $use_gpu \
      --time-in-secs true --det-window 5 \
      data/$name $face_det_model $face_reco_model $face_embed_ref_dir/$name
  done
fi


if [ $stage -le 3 ];then 
  #for name in sre19_av_v_dev_test
  for name in sre19_av_v_dev_enroll \
		sre19_av_v_dev_test \
		sre19_av_v_eval_enroll \
		sre19_av_v_eval_test \
		janus_dev_enroll \
		janus_dev_test_core \
		janus_eval_enroll \
		janus_eval_test_core 
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 40 ? $num_spk:40))
    steps_insightface/extract_face_embed_with_face_det_v4.sh \
    --cmd "$cmd" --nj $nj --use-gpu $use_gpu \
    data/$name $face_det_model $face_reco_model $face_embed_facedet_dir/$name
  done
fi

