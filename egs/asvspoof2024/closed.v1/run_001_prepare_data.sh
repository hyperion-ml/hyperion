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


if [ $stage -le 5 ];then
  echo "Prepare the ASV Spoof 2024 train dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset train \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_train

  echo "Prepare the ASV Spoof 2024 dev-enroll dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset dev_enroll \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_dev_enroll

  echo "Prepare the ASV Spoof 2024 dev-test dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset dev \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_dev
  
fi

