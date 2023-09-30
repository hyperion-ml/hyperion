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
  # Prepare the VoxCeleb2 dataset for training.
  hyperion-prepare-data voxceleb2 --subset dev --corpus-dir $voxceleb2_root \
			--cat-videos --use-kaldi-ids \
			--output-dir data/voxceleb2cat_train
fi

if [ $stage -le 2 ];then
  # prepare voxceleb1 for test
  hyperion-prepare-data voxceleb1 --task test --corpus-dir $voxceleb1_root \
			--use-kaldi-ids \
			--output-dir data/voxceleb1_test
fi

if [ $stage -le 3 ] && [ "$do_voxsrc22" == "true" ];then
  hyperion-prepare-data voxsrc22 --subset dev --corpus-dir $voxsrc22_root \
			--vox1-corpus-dir $voxceleb1_root \
			--output-dir data/voxsrc22_dev
fi

# if [ $stage -le 4 ] && [ "$do_voxsrc22" == "true" ];then
  #   hyperion-prepare-data voxsrc22 --subset test --corpus-dir $voxsrc22_root \
  # 		  --vox1-corpus-dir $voxceleb1_root \
  # 		  --output-dir data/voxsrc22_test
# fi

if [ $stage -le 5 ] && [ "$do_qmf" == "true" ];then
  # split vox2 into 2 parts, for cohort and qmf training
  hyperion-split-dataset-into-trials-and-cohort --data-dir data/voxceleb2cat_train
fi