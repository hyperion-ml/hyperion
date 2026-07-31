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
. ./datapath.sh 


nj=6

mkdir -p data


if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500
    do
      # use underscore-separated names in data directories.
      local/data_prep.sh ${librispeech_root}/${part} data/${part//-/_}
      steps_xvec/audio_to_duration.sh --cmd "$train_cmd" data/${part//-/_}
    done
fi

# if [ $stage -le 1 ]; then
#   echo "Stage 1: Prepare LibriSpeech manifest"
#   # We assume that you have downloaded the LibriSpeech corpus
#   # to $librispeech_root
#   mkdir -p data/manifests
#   if [ ! -e data/manifests/.librispeech.done ]; then
#     lhotse prepare librispeech -j $nj $librispeech_root data/manifests
#     touch data/manifests/.librispeech.done
#   fi
# fi

# if [ $stage -le 2 ]; then
#   echo "Stage 2: Prepare musan manifest"
#   # We assume that you have downloaded the musan corpus
#   # to $musan_root
#   mkdir -p data/manifests
#   if [ ! -e data/manifests/.musan.done ]; then
#     lhotse prepare musan $musan_root data/manifests
#     touch data/manifests/.musan.done
#   fi
# fi
