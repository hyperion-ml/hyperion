#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
# ---------------------------------------------------------------------------
# run_001_prepare_data.sh
# ---------------------------------------------------------------------------
# Stage 1 of the VoxCeleb v1.2 recipe. It prepares the data manifests,
# invoking Hyperion helpers to create the Kaldi-style directories for
# VoxCeleb2 (train), VoxCeleb1 (test) and VoxSRC when enabled. You can
# control which blocks execute by raising/lowering the `stage` variable
# or by toggling the `do_voxsrc22` / `do_qmf` flags in the config files.
# ---------------------------------------------------------------------------
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 
. $config_file

if [ $stage -le 1 ];then
  echo "[run001][stage1] Preparing VoxCeleb2 cat training set"
  hyperion-prepare-data voxceleb2 --subset dev --corpus-dir $voxceleb2_root \
		--cat-videos --use-kaldi-ids \
		--output-dir data/voxceleb2cat_train
fi

if [ $stage -le 2 ];then
  echo "[run001][stage2] Preparing VoxCeleb1 test set"
  hyperion-prepare-data voxceleb1 --task test --corpus-dir $voxceleb1_root \
		--use-kaldi-ids \
		--output-dir data/voxceleb1_test
fi

if [ $stage -le 3 ] && [ "$do_voxsrc22" == "true" ];then
  echo "[run001][stage3] Preparing VoxSRC22 dev set"
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
  echo "[run001][stage5] Splitting VoxCeleb2 for cohort/QMF"
  hyperion-split-dataset-into-trials-and-cohort --data-dir data/voxceleb2cat_train
fi
