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

. parse_options.sh || exit 1;
. datapath.sh 
. $config_file

if [ $stage -le 1 ];then
  echo "Prepare SRE CTS Superset"
  hyperion-prepare-data sre_cts_superset \
			--corpus-dir $sre_superset_root \
			--use-kaldi-ids \
			--output-dir data/sre_cts_superset
fi

if [ $stage -le 2 ];then
  echo "Prepare SRE16"
  hyperion-prepare-data sre16 \
			--corpus-dir $sre16_root \
			--subset eval \
			--partition train \
			--output-dir data/sre16_eval_train
fi

if [ $stage -le 3 ];then
  echo "Prepare SRE21 dev"
  hyperion-prepare-data sre21 \
			--corpus-dir $sre21_dev_root \
			--modality audio \
			--subset dev \
			--partition enrollment \
			--use-ldc-langs \
			--output-dir data/sre21_audio_dev_enroll

  hyperion-prepare-data sre21 \
			--corpus-dir $sre21_dev_root \
			--modality audio \
			--subset dev \
			--partition test \
			--use-ldc-langs \
			--output-dir data/sre21_audio_dev_test

  hyperion-prepare-data sre21 \
			--corpus-dir $sre21_dev_root \
			--modality audio-visual \
			--subset dev \
			--partition test \
			--use-ldc-langs \
			--target-sample-freq 16000 \
			--output-dir data/sre21_audio-visual_dev_test

fi

if [ $stage -le 4 ];then
  echo "Prepare SRE21 eval"
  hyperion-prepare-data sre21 \
			--corpus-dir $sre21_eval_root \
			--modality audio \
			--subset eval \
			--partition enrollment \
			--use-ldc-langs \
			--output-dir data/sre21_audio_eval_enroll

  hyperion-prepare-data sre21 \
			--corpus-dir $sre21_eval_root \
			--modality audio \
			--subset eval \
			--partition test \
			--use-ldc-langs \
			--output-dir data/sre21_audio_eval_test

  hyperion-prepare-data sre21 \
			--corpus-dir $sre21_eval_root \
			--modality audio-visual \
			--subset eval \
			--partition test \
			--use-ldc-langs \
			--target-sample-freq 16000 \
			--output-dir data/sre21_audio-visual_eval_test
fi


if [ $stage -le 9 ];then
  echo "Prepare SRE24 dev"
  hyperion-prepare-data sre24 \
			--corpus-dir $sre24_dev_root \
			--corpus-docs-dir $sre24_dev_docs_root \
			--modality audio \
			--subset dev \
			--partition enrollment \
			--use-ldc-langs \
			--output-dir data/sre24_audio_dev_enroll

  hyperion-prepare-data sre24 \
			--corpus-dir $sre24_dev_root \
			--corpus-docs-dir $sre24_dev_docs_root \
			--modality audio \
			--subset dev \
			--partition test \
			--use-ldc-langs \
			--output-dir data/sre24_audio_dev_test

  hyperion-prepare-data sre24 \
			--corpus-dir $sre24_dev_root \
			--corpus-docs-dir $sre24_dev_docs_root \
			--modality audio-visual \
			--subset dev \
			--partition test \
			--use-ldc-langs \
			--target-sample-freq 16000 \
			--output-dir data/sre24_audio-visual_dev_test
  
fi


if [ $stage -le 10 ];then
  echo "Prepare SRE24 eval"
  hyperion-prepare-data sre24 \
			--corpus-dir $sre24_eval_root \
			--modality audio \
			--subset eval \
			--partition enrollment \
			--use-ldc-langs \
			--output-dir data/sre24_audio_eval_enroll

  hyperion-prepare-data sre24 \
			--corpus-dir $sre24_eval_root \
			--modality audio \
			--subset eval \
			--partition test \
			--use-ldc-langs \
			--output-dir data/sre24_audio_eval_test

  hyperion-prepare-data sre24 \
			--corpus-dir $sre24_eval_root \
			--modality audio-visual \
			--subset eval \
			--partition test \
			--use-ldc-langs \
			--target-sample-freq 16000 \
			--output-dir data/sre24_audio-visual_eval_test
  
fi

exit

# if [ $stage -le 5 ];then
#   echo "Prepare Janus Dev"
#   hyperion-prepare-data janus_multimedia \
# 			--corpus-dir $janus_root \
# 			--subset dev \
# 			--partition enrollment \
# 			--target-sample-freq 16000 \
# 			--output-dir data/janus_dev_enroll

#   hyperion-prepare-data janus_multimedia \
# 			--corpus-dir $janus_root \
# 			--subset dev \
# 			--condition core \
# 			--partition test \
# 			--target-sample-freq 16000 \
# 			--output-dir data/janus_dev_core_test

#   hyperion-prepare-data janus_multimedia \
# 			--corpus-dir $janus_root \
# 			--subset dev \
# 			--condition full \
# 			--partition test \
# 			--target-sample-freq 16000 \
# 			--output-dir data/janus_dev_full_test

# fi

# if [ $stage -le 6 ];then
#   echo "Prepare Janus Eval"
#   hyperion-prepare-data janus_multimedia \
# 			--corpus-dir $janus_root \
# 			--subset eval \
# 			--partition enrollment \
# 			--target-sample-freq 16000 \
# 			--output-dir data/janus_eval_enroll

#   hyperion-prepare-data janus_multimedia \
# 			--corpus-dir $janus_root \
# 			--subset eval \
# 			--condition core \
# 			--partition test \
# 			--target-sample-freq 16000 \
# 			--output-dir data/janus_eval_core_test

#   hyperion-prepare-data janus_multimedia \
# 			--corpus-dir $janus_root \
# 			--subset eval \
# 			--condition full \
# 			--partition test \
# 			--target-sample-freq 16000 \
# 			--output-dir data/janus_eval_full_test

# fi

