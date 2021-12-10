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
  for name in voxcelebcat \
		voxcelebcat_8k \
  		sre_cts_superset_16k_trn \
  		sre16_eval_tr60_tgl \
  		sre16_eval_tr60_yue \
  		sre16_train_dev_ceb \
  		sre16_train_dev_cmn
  do
    steps_xvec/preprocess_audios_for_nnet_train.sh \
      --nj 40 --cmd "$train_cmd" \
      --storage_name sre21-v1.16k-$(date +'%m_%d_%H_%M') --use-bin-vad true \
      data/${name} data/${name}_proc_audio_no_sil exp/${name}_proc_audio_no_sil
    utils/fix_data_dir.sh data/${name}_proc_audio_no_sil
  done
fi

if [ $stage -le 2 ];then
  utils/combine_data.sh \
    data/voxcelebcat_sre_alllangs_mixfs_proc_audio_no_sil \
    data/voxcelebcat_proc_audio_no_sil \
    data/voxcelebcat_8k_proc_audio_no_sil \
    data/sre_cts_superset_16k_trn_proc_audio_no_sil \
    data/sre16_eval_tr60_tgl_proc_audio_no_sil \
    data/sre16_eval_tr60_yue_proc_audio_no_sil \
    data/sre16_train_dev_ceb_proc_audio_no_sil \
    data/sre16_train_dev_cmn_proc_audio_no_sil
fi

if [ $stage -le 3 ]; then
    # Now, we remove files with less than 4s
    hyp_utils/remove_short_audios.sh --min-len 4 data/${nnet_data}_proc_audio_no_sil

    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 2 utterances.
    hyp_utils/remove_spk_few_utts.sh --min-num-utts 2 data/${nnet_data}_proc_audio_no_sil

fi

if [ $stage -le 4 ]; then
    # Prepare train and validation lists for x-vectors
    local/make_train_lists_sup_embed_with_augm.sh \
	data/${nnet_data}_proc_audio_no_sil \
	data/${nnet_data}_proc_audio_no_sil/lists_xvec
fi

exit
