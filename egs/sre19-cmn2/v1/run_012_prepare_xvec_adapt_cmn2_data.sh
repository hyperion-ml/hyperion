#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

nnet_adapt_data=sre18_cmn2_adapt_lab_combined

export TMPDIR=data/tmp
mkdir -p $TMPDIR

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 2 ]; then
    # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
    # wasteful, as it roughly doubles the amount of training data on disk.  After
    # creating training examples, this can be removed.
    steps_kaldi_xvec/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd -l \"hostname=b[01]*\" -V" \
	--storage_name sre19-cmn2-v1-$(date +'%m_%d_%H_%M') \
	data/${nnet_adapt_data} data/${nnet_adapt_data}_no_sil exp/${nnet_adapt_data}_no_sil
    utils/fix_data_dir.sh data/${nnet_adapt_data}_no_sil

fi


if [ $stage -le 3 ]; then
    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 4s (400 frames) per utterance.
    hyp_utils/remove_short_utts.sh --min-len 400 data/${nnet_adapt_data}_no_sil

    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 2 utterances.
    hyp_utils/remove_spk_few_utts.sh --min-num-utts 2 data/${nnet_adapt_data}_no_sil
    
fi

if [ $stage -le 4 ];then
    #get english data for 376 spk (2 x 188)
    mkdir -p data/${nnet_data}_376spk_no_sil
    cp data/${nnet_data}_no_sil/* data/${nnet_data}_376spk_no_sil 2>/dev/null
    #choose spk from sre12
    awk '{ print $1}' data/sre12_tel/spk2utt | head -n 410 > data/${nnet_data}_376spk_no_sil/spk_list
    utils/filter_scp.pl -f 2 data/${nnet_data}_376spk_no_sil/spk_list \
	data/${nnet_data}_no_sil/utt2spk > \
	data/${nnet_data}_376spk_no_sil/utt2spk
    utils/fix_data_dir.sh data/${nnet_data}_376spk_no_sil
    
fi

if [ $stage -le 5 ];then
  utils/combine_data.sh --extra-files "utt2num_frames" \
      data/adapt_combined_no_sil \
      data/${nnet_data}_376spk_no_sil data/${nnet_adapt_data}_no_sil
    
fi

exit
