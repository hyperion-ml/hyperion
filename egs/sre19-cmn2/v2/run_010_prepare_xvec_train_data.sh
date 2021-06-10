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
. $config_file

export TMPDIR=data/tmp
mkdir -p $TMPDIR

if [ $stage -le 1 ];then
  # Combine data to train x-vector nnet
  utils/combine_data.sh --extra-files "utt2num_frames" \
      data/train_combined \
      data/swbd_sre_tel_combined data/voxcelebcat_tel_combined 
fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 2 ]; then
    # This script applies CMVN and removes nonspeech frames.  
    steps_xvec/prepare_feats_for_nnet_train.sh --nj 40 --cmd "$train_cmd" \
	--storage_name train-sre19-cmn2-$(date +'%m_%d_%H_%M') --nodes c0 \
	data/${nnet_data} data/${nnet_data}_no_sil exp/${nnet_data}_no_sil
    utils/fix_data_dir.sh data/${nnet_data}_no_sil

fi


if [ $stage -le 3 ]; then
    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 4s (400 frames) per utterance.
    hyp_utils/remove_short_utts.sh --min-len 400 data/${nnet_data}_no_sil

    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 8 utterances.
    hyp_utils/remove_spk_few_utts.sh --min-num-utts 8 data/${nnet_data}_no_sil
fi

if [ $stage -le 4 ]; then
    # Prepare train and validation lists for x-vectors
    local/make_train_lists_sup_embed_with_augm.sh data/${nnet_data}_no_sil data/${nnet_data}_no_sil/lists_xvec
fi

exit
