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


# if [ ${stage} -le 1 ]; then
#     ### Task dependent. You have to make data the following preparation part by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 0: Data preparation"
#     for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500
#     do
#       # use underscore-separated names in data directories.
#       local/data_prep.sh ${librispeech_root}/${part} data/${part//-/_}
#       steps_xvec/audio_to_duration.sh --cmd "$train_cmd" data/${part//-/_}
#     done
# fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Prepare lhotse LibriSpeech manifest"
  # We assume that you have downloaded the LibriSpeech corpus
  # to $librispeech_root
  mkdir -p data/lhotse_librispeech
  if [ ! -e data/lhotse_librispeech/.librispeech.done ]; then
    lhotse prepare librispeech -j $nj $librispeech_root data/lhotse_librispeech
    touch data/lhotse_librispeech/.librispeech.done
  fi
fi

if [ $stage -le 2 ];then
  echo "Stage 2: Convert Manifest to Hyperion Datasets"
  for data in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other
  do
    hyperion-dataset from_lhotse \
		     --recordings-file data/lhotse_librispeech/librispeech_recordings_${data}.jsonl.gz \
		     --supervisions-file data/lhotse_librispeech/librispeech_supervisions_${data}.jsonl.gz \
		     --dataset data/librispeech_${data}
  done
    
fi

if [ $stage -le 3 ];then
  echo "Stage 3: Merge Librispeech train sets"
  hyperion-dataset merge \
		   --input-datasets data/librispeech_train-{clean-100,clean-360,other-500} \
		   --dataset data/librispeech_train-960

  echo "Stage 3: Merge Librispeech dev sets"
  hyperion-dataset merge \
		   --input-datasets data/librispeech_dev-{clean,other} \
		   --dataset data/librispeech_dev

fi
