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

nnet_data=$spknet_data

if [ $stage -le 2 ]; then
    # This script preprocess audio for x-vector training
    steps_xvec/preprocess_audios_for_nnet_train.sh --nj 40 --cmd "$train_cmd" \
	--storage_name voxceleb-adv.v2-$(date +'%m_%d_%H_%M') --use-bin-vad true \
	data/${nnet_data} data/${nnet_data}_proc_audio_no_sil exp/${nnet_data}_proc_audio_no_sil
    utils/fix_data_dir.sh data/${nnet_data}_proc_audio_no_sil

fi

if [ $stage -le 3 ]; then
    # Now, we remove files with less than 4s
    hyp_utils/remove_short_audios.sh --min-len 4 data/${nnet_data}_proc_audio_no_sil

    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 4 utterances.
    hyp_utils/remove_spk_few_utts.sh --min-num-utts 4 data/${nnet_data}_proc_audio_no_sil

fi

if [ $stage -le 4 ]; then
    # Prepare train and validation lists for x-vectors
    local/make_train_lists_sup_embed_with_augm.sh \
	data/${nnet_data}_proc_audio_no_sil \
	data/${nnet_data}_proc_audio_no_sil/lists_xvec
fi

exit
