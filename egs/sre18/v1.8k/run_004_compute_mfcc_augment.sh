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
mfccdir=`pwd`/mfcc

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;

if [ $stage -le 1 ];then
    
  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  for name in swbd_sre_tel_aug_170k sre_phnmic_aug_20k voxceleb_aug_250k 
  do
      steps/make_mfcc.sh --mfcc-config conf/mfcc_8k.conf --nj 120 --cmd "$train_cmd" \
      			 data/$name exp/make_mfcc $mfccdir
      fix_data_dir.sh data/$name
  done

fi


if [ $stage -le 2 ];then
    
    # Combine the clean and augmented lists.  
    utils/combine_data.sh data/swbd_sre_tel_combined data/swbd_sre_tel_aug_170k data/swbd_sre_tel
    utils/combine_data.sh data/sre_phnmic_combined data/sre_phnmic_aug_20k data/sre_phnmic
    utils/combine_data.sh data/voxceleb_combined data/voxceleb_aug_250k data/voxceleb

    # Filter out the clean + augmented portion of the SRE list.  
    utils/copy_data_dir.sh data/swbd_sre_tel_combined data/sre_tel_combined
    utils/filter_scp.pl data/sre_tel/spk2utt data/swbd_sre_tel_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_tel_combined/utt2spk
    utils/fix_data_dir.sh data/sre_tel_combined
  
fi
    
exit
