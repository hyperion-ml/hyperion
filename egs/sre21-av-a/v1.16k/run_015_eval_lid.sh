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
xvec_chunk_length=12800
ft=0
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
    xvec_args="--chunk-length $xvec_chunk_length"
fi

lid_nnet_name=lresnet34_lid_v1
lid_logits_dir=exp/lid_logits/$lid_nnet_name
nnet=exp/lid_nnets/lresnet34_lid_v1/model_ep0070.pth

if [ $stage -le 1 ]; then
  for name in voxcelebcat_8k \
		sre21_audio_dev_enroll \
		sre21_audio_dev_test \
		sre21_audio-visual_dev_test \
		sre21_audio_eval_enroll \
		sre21_audio_eval_test \
		sre21_audio-visual_eval_test
  do
    steps_xvec/eval_xvec_logits_from_wav.sh \
      --cmd "$xvec_cmd --mem 12G" --nj 100 ${xvec_args} \
      --feat-config conf/fbank64_stmn_nb_16k.yaml \
      $nnet data/${name} \
      $lid_logits_dir/${name}
  done
fi


if [ $stage -le 2 ];then
  train_list_dir=data/train_lid_proc_audio_no_sil
  for name in voxcelebcat_8k
  do
    hyp_utils/conda_env.sh \
      local/estimate_lid_labels.py \
      --list-file data/$name/utt2spk \
      --logits-file scp:$lid_logits_dir/$name/logits.scp \
      --class-file $train_list_dir/lists_train_lid/class2int \
      --output-file data/$name/utt2est_lang
  done
  
  for name in sre21_audio_dev_enroll \
		sre21_audio_dev_test \
		sre21_audio-visual_dev_test \
		sre21_audio_eval_enroll \
		sre21_audio_eval_test \
		sre21_audio-visual_eval_test
  do
    echo "Estimating language labels for $name"
    hyp_utils/conda_env.sh \
      local/estimate_lid_labels.py \
      --list-file data/$name/utt2spk \
      --logits-file scp:$lid_logits_dir/$name/logits.scp \
      --class-file $train_list_dir/lists_train_lid/class2int \
      --output-file data/$name/utt2est_lang \
      --sre21
  done

  awk '{ sub(/-8k$/,"",$1); print $0}' \
      data/voxcelebcat_8k/utt2est_lang \
      > data/voxcelebcat/utt2est_lang
  
fi

if [ $stage -le 3 ];then

  rm -f lid_{cts,afv}_{pred,gt}
  for name in sre21_audio_dev_enroll \
		sre21_audio_dev_test \
		sre21_audio-visual_dev_test
  do
    awk '$1~/\.flac$/ || $1 ~/\.mp4$/' data/$name/utt2lang >> lid_afv_gt
    awk '$1~/\.flac$/ || $1 ~/\.mp4$/' data/$name/utt2est_lang >> lid_afv_pred
    awk '$1~/\.sph$/' data/$name/utt2lang >> lid_cts_gt
    awk '$1~/\.sph$/' data/$name/utt2est_lang >> lid_cts_pred

  done
  echo "Acc SRE21 dev AFV"
  hyp_utils/conda_env.sh \
    local/compute_lid_acc.py \
    --pred-file lid_afv_pred \
    --gt-file lid_afv_gt
  
  echo "Acc SRE21 dev CTS"
  hyp_utils/conda_env.sh \
    local/compute_lid_acc.py \
    --pred-file lid_cts_pred \
    --gt-file lid_cts_gt \

  rm -f lid_{cts,afv}_{pred,gt}
fi


