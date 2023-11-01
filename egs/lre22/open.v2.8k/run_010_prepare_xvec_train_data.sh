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

. parse_options.sh || exit 1;
. $config_file

if [ $stage -le 1 ]; then
  # This script preprocess audio for x-vector training
  for name in voxlingua107_codecs \
		lre17_train \
  		lre17_{dev,eval}_{cts,afv,afv_codecs} \
		babel sre16-21_cts sre_cts_superset \
		sre21_afv_codecs cv_codecs adi17_codecs \
		lwazi09{,_codecs} nchlt14{,_codecs} fleurs22{,_codecs} ammi20{,_codecs} ast{,_codecs}
  do
    steps_xvec/preprocess_audios_for_nnet_train.sh \
      --nj 40 --cmd "$train_cmd" \
      --storage_name lre22-fixed-v1.8k-$(date +'%m_%d_%H_%M') --use-bin-vad true \
      data/${name} data/${name}_proc_audio_no_sil exp/${name}_proc_audio_no_sil
    utils/fix_data_dir.sh data/${name}_proc_audio_no_sil
  done
fi

if [ $stage -le 2 ];then
  utils/combine_data.sh \
    data/lre17_proc_audio_no_sil \
    data/lre17_train_proc_audio_no_sil \
    data/lre17_{dev,eval}_{cts,afv,afv_codecs}_proc_audio_no_sil

  utils/combine_data.sh \
    data/babel_sre_proc_audio_no_sil \
    data/{babel,sre16-21_cts,sre21_afv_codecs,sre_cts_superset}_proc_audio_no_sil

  utils/combine_data.sh \
    data/others_afr_proc_audio_no_sil \
    data/adi17_proc_audio_no_sil \
    data/{lwazi09,nchlt14,fleurs22,ammi20,ast}{,_codecs}_proc_audio_no_sil
fi

if [ $stage -le 3 ]; then
  # Now, we remove files with less than 3s
  hyp_utils/remove_short_audios.sh --min-len 3 data/voxlingua107_codecs_proc_audio_no_sil
  hyp_utils/remove_short_audios.sh --min-len 3 data/lre17_proc_audio_no_sil
  hyp_utils/remove_short_audios.sh --min-len 3 data/babel_sre_proc_audio_no_sil
  hyp_utils/remove_short_audios.sh --min-len 3 data/others_afr_proc_audio_no_sil
  hyp_utils/remove_short_audios.sh --min-len 3 data/cv_codecs_proc_audio_no_sil
fi

if [ $stage -le 4 ];then
  # merge all data
  utils/combine_data.sh \
    data/open_proc_audio_no_sil \
    data/{voxlingua107_codecs,lre17,babel_sre,cv_codecs,others_afr}_proc_audio_no_sil \
fi


if [ $stage -le 5 ]; then
  for name in open_proc_audio_no_sil
  do
    hyp_utils/conda_env.sh \
      local/split_segments_train_val.py \
      --segments-file data/$name/utt2lang \
      --recordings-file data/$name/wav.scp \
      --durations-file data/$name/utt2dur \
      --val-percent 2. \
      --remove-langs fra-mix ara-ary en-en es-es pt-pt ar-ar \
      --output-dir data/$name/train_val_split
  done
fi
