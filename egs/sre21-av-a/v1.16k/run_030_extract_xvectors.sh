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
xvec_chunk_length=5000
ft=1
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
    xvec_args="--chunk-length $xvec_chunk_length"
fi

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
    nnet=$ft_nnet
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
    nnet=$ft2_nnet
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
    nnet=$ft3_nnet
fi

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ]; then
  # Extract xvectors for training LDA/PLDA on Audio from Video Data
  # we sample 10-60 sec chunks 
  for name in voxcelebcat voxcelebcat_8k
  do
    if [ $plda_num_augs -eq 0 ]; then
      steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 12G" --nj 100 ${xvec_args} \
	--random-utt-length true --min-utt-length 1000 --max-utt-length 6000 \
	--feat-config $feat_config \
    	$nnet data/${name} \
    	$xvector_dir/${name}
    else
      steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 12G" --nj 300 ${xvec_args} \
	--random-utt-length true --min-utt-length 1000 --max-utt-length 6000 \
	--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    	$nnet data/${name} \
    	$xvector_dir/${name}_augx${plda_num_augs} \
	data/${name}_augx${plda_num_augs}
    fi
  done
fi

if [ $stage -le 2 ]; then
  # Extracts x-vectors for telephone datasets
  # This datasets already have speech durations between 10-60secs so we don't do any subsampling
  for name in sre_cts_superset_16k_trn \
		sre16_eval_tr60_tgl \
  		sre16_eval_tr60_yue \
		sre16_train_dev_ceb \
  		sre16_train_dev_cmn
  do
    num_utts=$(wc -l data/$name/wav.scp | awk '{ print $1}')
    if [ $plda_num_augs -eq 0 ]; then
      nj=$(($num_utts < 300 ? $num_utts:300))
      steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 12G" --nj $nj ${xvec_args} \
	--feat-config $feat_config \
    	$nnet data/${name} \
    	$xvector_dir/${name}
    else
      nj=$(($num_utts < 1000 ? $num_utts:1000))
      steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 12G" --nj $nj ${xvec_args} \
	--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    	$nnet data/${name} \
    	$xvector_dir/${name}_augx${plda_num_augs} \
	data/${name}_augx${plda_num_augs}
    fi
  done
fi

if [ $stage -le 3 ]; then
    # Extracts x-vectors for evaluation
  for name in sre16_eval40_yue_enroll \
  		sre16_eval40_yue_test \
		sre_cts_superset_16k_dev \
		sre21_audio_dev_enroll \
		sre21_audio_dev_test \
		sre21_audio-visual_dev_test \
	      	sre21_audio_eval_enroll \
		sre21_audio_eval_test \
		sre21_audio-visual_eval_test
  do
    num_utts=$(wc -l data/$name/wav.scp | awk '{ print $1}')
    nj=$(($num_utts < 100 ? $num_utts:100))
    steps_xvec/extract_xvectors_from_wav.sh \
      --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
      --feat-config $feat_config \
      $nnet data/$name \
      $xvector_dir/$name
  done
fi

if [ $stage -le 4 ]; then
  # merge eval x-vectors lists
  mkdir -p $xvector_dir/sre16_eval40_yue
  cat $xvector_dir/sre16_eval40_yue_{enroll,test}/xvector.scp > $xvector_dir/sre16_eval40_yue/xvector.scp

  mkdir -p $xvector_dir/sre21_audio_dev
  cat $xvector_dir/sre21_audio_dev_{enroll,test}/xvector.scp > $xvector_dir/sre21_audio_dev/xvector.scp

  mkdir -p $xvector_dir/sre21_audio-visual_dev
  cat $xvector_dir/sre21_{audio_dev_enroll,audio-visual_dev_test}/xvector.scp > $xvector_dir/sre21_audio-visual_dev/xvector.scp

  mkdir -p $xvector_dir/sre21_audio_eval
  cat $xvector_dir/sre21_audio_eval_{enroll,test}/xvector.scp > $xvector_dir/sre21_audio_eval/xvector.scp

  mkdir -p $xvector_dir/sre21_audio-visual_eval
  cat $xvector_dir/sre21_{audio_eval_enroll,audio-visual_eval_test}/xvector.scp > $xvector_dir/sre21_audio-visual_eval/xvector.scp

fi

if [ $stage -le 5 ];then
  # merge training datasets
  utils/combine_data.sh \
    data/sre_alllangs \
    data/sre_cts_superset_16k_trn \
    data/sre16_eval_tr60_tgl \
    data/sre16_eval_tr60_yue \
    data/sre16_train_dev_ceb \
    data/sre16_train_dev_cmn

  mkdir -p $xvector_dir/sre_alllangs
  for name in sre_cts_superset_16k_trn \
    sre16_eval_tr60_tgl \
    sre16_eval_tr60_yue \
    sre16_train_dev_ceb \
    sre16_train_dev_cmn
  do
    cat $xvector_dir/$name/xvector.scp
  done > $xvector_dir/sre_alllangs/xvector.scp

  utils/combine_data.sh \
    data/voxceleb_sre_alllangs_8k \
    data/voxcelebcat_8k \
    data/sre_alllangs

  mkdir -p $xvector_dir/voxceleb_sre_alllangs_8k
  cat $xvector_dir/{voxcelebcat_8k,sre_alllangs}/xvector.scp \
      > $xvector_dir/voxceleb_sre_alllangs_8k/xvector.scp

  utils/combine_data.sh \
    data/voxceleb_sre_alllangs_mixfs \
    data/voxcelebcat \
    data/voxceleb_sre_alllangs_8k

  mkdir -p $xvector_dir/voxceleb_sre_alllangs_mixfs
  cat $xvector_dir/{voxcelebcat,voxceleb_sre_alllangs_8k}/xvector.scp \
      > $xvector_dir/voxceleb_sre_alllangs_mixfs/xvector.scp

fi

# if [ $stage -le 5 ]; then

#     utils/combine_data.sh data/sitw_sre18_dev_vast data/sitw_dev data/sre18_dev_vast
#     mkdir -p $xvector_dir/sitw_sre18_dev_vast
#     cat $xvector_dir/{sitw_dev,sre18_dev_vast}/xvector.scp > $xvector_dir/sitw_sre18_dev_vast/xvector.scp
# fi

