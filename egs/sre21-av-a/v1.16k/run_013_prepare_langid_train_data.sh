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

if [ $stage -le 1 ];then
  utils/combine_data.sh \
    data/train_lid_proc_audio_no_sil \
    data/sre_cts_superset_16k_trn_proc_audio_no_sil \
    data/sre16_eval_tr60_tgl_proc_audio_no_sil \
    data/sre16_eval_tr60_yue_proc_audio_no_sil \
    data/sre16_train_dev_ceb_proc_audio_no_sil \
    data/sre16_train_dev_cmn_proc_audio_no_sil
fi

if [ $stage -le 2 ]; then
    # Now, we remove files with less than 4s
    hyp_utils/remove_short_audios.sh --min-len 4 data/train_lid_proc_audio_no_sil
fi

if [ $stage -le 3 ]; then
    # Prepare train and validation lists for x-vectors
    local/make_train_lists_lang_embed.sh \
	data/train_lid_proc_audio_no_sil \
	data/train_lid_proc_audio_no_sil/lists_train_lid
fi

exit
