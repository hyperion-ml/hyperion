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

. parse_options.sh || exit 1;


if [ $stage -le 1 ];then
  # Combine data to train x-vector nnet
  hyp_utils/combine_data.sh data/train_combined data/sitw_train_combined data/voxceleb_combined 

fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 2 ]; then
    # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
    # wasteful, as it roughly doubles the amount of training data on disk.  After
    # creating training examples, this can be removed.
    steps_kaldi_xvec/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd -l \"hostname=b[01]*\" -V" \
					      --storage_name voices_challenge-v1-$(date +'%m_%d_%H_%M') \
					      data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
    hyp_utils/fix_data_dir.sh data/train_combined_no_sil

fi


if [ $stage -le 3 ]; then
    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 4s (400 frames) per utterance.
    min_len=400
    mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
    awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
    utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
    mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
    hyp_utils/fix_data_dir.sh data/train_combined_no_sil
    
    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 8 utterances.
    min_num_utts=8
    awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
    awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
    mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
    utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk
    
    utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
    mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames
    
    # Now we're ready to create training examples.
    hyp_utils/fix_data_dir.sh data/train_combined_no_sil

fi


exit
