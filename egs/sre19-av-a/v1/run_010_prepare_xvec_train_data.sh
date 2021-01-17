#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 1 ]; then
    # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
    # wasteful, as it roughly doubles the amount of training data on disk.  After
    # creating training examples, this can be removed.
    steps_kaldi_xvec/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
	--storage_name sre19-av-v1-$(date +'%m_%d_%H_%M') \
	data/${nnet_data} data/${nnet_data}_no_sil exp/${nnet_data}_no_sil
    utils/fix_data_dir.sh data/${nnet_data}_no_sil

fi


if [ $stage -le 2 ]; then
    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 4s (400 frames) per utterance.
    hyp_utils/remove_short_utts.sh --min-len 400 data/${nnet_data}_no_sil
    
    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 8 utterances.
    hyp_utils/remove_spk_few_utts.sh --min-num-utts 8 data/${nnet_data}_no_sil
 
fi


exit
