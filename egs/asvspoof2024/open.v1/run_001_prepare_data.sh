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
  # Prepare the ASV Spoof 2015 dataset for training.
  hyperion-prepare-data asvspoof2015 \
  			--subset train \
  			--corpus-dir $asvspoof2015_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2015_train
  hyperion-prepare-data asvspoof2015 \
  			--subset dev \
  			--corpus-dir $asvspoof2015_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2015_dev
  hyperion-prepare-data asvspoof2015 \
			--subset eval \
			--corpus-dir $asvspoof2015_root \
			--use-kaldi-ids \
			--output-dir data/asvspoof2015_eval

fi

if [ $stage -le 2 ];then
  # Prepare the ASV Spoof 2017 dataset for training.
  hyperion-prepare-data asvspoof2017 \
  			--subset train \
  			--corpus-dir $asvspoof2017_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2017_train
  hyperion-prepare-data asvspoof2017 \
  			--subset dev \
  			--corpus-dir $asvspoof2017_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2017_dev
  hyperion-prepare-data asvspoof2017 \
			--subset eval \
			--corpus-dir $asvspoof2017_root \
			--use-kaldi-ids \
			--output-dir data/asvspoof2017_eval

fi

if [ $stage -le 3 ];then
  echo "Prepare the ASV Spoof 2019 LA dataset"
  hyperion-prepare-data asvspoof2019 \
			--subset la_train \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_train
  hyperion-prepare-data asvspoof2019 \
			--subset la_dev \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_dev
  hyperion-prepare-data asvspoof2019 \
			--subset la_eval \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_eval
  hyperion-prepare-data asvspoof2019 \
			--subset la_dev_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_dev_enroll
  hyperion-prepare-data asvspoof2019 \
			--subset la_eval_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_eval_enroll

  echo "Prepare the ASV Spoof 2019 PA dataset"
  hyperion-prepare-data asvspoof2019 \
			--subset pa_train \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_train
  hyperion-prepare-data asvspoof2019 \
			--subset pa_dev \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_dev
  hyperion-prepare-data asvspoof2019 \
			--subset pa_eval \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_eval
  hyperion-prepare-data asvspoof2019 \
			--subset pa_dev_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_dev_enroll
  hyperion-prepare-data asvspoof2019 \
			--subset pa_eval_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_eval_enroll

fi

if [ $stage -le 4 ];then
  echo "Prepare the ASV Spoof 2021 LA dataset"
  hyperion-prepare-data asvspoof2021 \
			--subset la_eval \
			--corpus-dir $asvspoof2021_root \
			--output-dir data/asvspoof2021_la_eval
  
  echo "Prepare the ASV Spoof 2021 DF dataset"
  hyperion-prepare-data asvspoof2021 \
			--subset df_eval \
			--corpus-dir $asvspoof2021_root \
			--output-dir data/asvspoof2021_df_eval

  echo "Prepare the ASV Spoof 2021 PA dataset"
  hyperion-prepare-data asvspoof2021 \
			--subset pa_eval \
			--corpus-dir $asvspoof2021_root \
			--output-dir data/asvspoof2021_pa_eval

fi

if [ $stage -le 5 ];then
  # echo "Prepare the ASV Spoof 2024 train dataset"
  # hyperion-prepare-data asvspoof2024 \
  # 			--subset train \
  # 			--corpus-dir $asvspoof2024_root \
  # 			--output-dir data/asvspoof2024_train

  # echo "Prepare the ASV Spoof 2024 dev-enroll dataset"
  # hyperion-prepare-data asvspoof2024 \
  # 			--subset dev_enroll \
  # 			--corpus-dir $asvspoof2024_root \
  # 			--output-dir data/asvspoof2024_dev_enroll

  echo "Prepare the ASV Spoof 2024 dev-test dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset dev \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_dev
  
fi

if [ $stage -le 6 ];then
  echo "Prepare the ASV Spoof 2024 progress dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset progress \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_prog

  # echo "Prepare the ASV Spoof 2024 progress-enroll dataset"
  # hyperion-prepare-data asvspoof2024 \
  # 			--subset progress_enroll \
  # 			--corpus-dir $asvspoof2024_root \
  # 			--output-dir data/asvspoof2024_prog_enroll

  
fi

