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
. $config_file


nj=6

mkdir -p data



if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Data preparation"
    for lan in $lans 
    do
      # use underscore-separated names in data directories.
      local/data_prep.sh ${lan} $commonvoice_root data/
    done
fi

if [ ${stage} -le 2 ]; then
  echo "stage 2: Data conversion"
  # for part in $test_data $dev_data $nnet_data
  for lan in $lans 
  do
    for part in ${lan}_test ${lan}_dev ${lan}_train
    do
      echo ${part}
      steps_transducer/preprocess_audios_for_nnet_train.sh --nj 16 --cmd "$train_cmd" \
      --storage_name commonvoice-v1-$(date +'%m_%d_%H_%M') --use-bin-vad false \
      --osr 16000 data/${part} data/${part}_proc_audio  exp/${part}_proc_audio
      utils/fix_data_dir.sh data/${part}_proc_audio || true
    done
  done
fi

if [ ${stage} -le 3 ]; then
  echo "stage 3: Combine Multilingual Data"
  
  dev_folders=""
  train_folders=""
  for lan in $lans 
  do
    dev_folders+="data/${lan}_dev_proc_audio "
    train_folders+="data/${lan}_train_proc_audio "
  done 
  
  combine_data.sh data/dev_data/ $dev_folders
  combine_data.sh data/nnet_data/ $train_folders


fi