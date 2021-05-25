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

. parse_options.sh || exit 1;

if [ $stage -le 1 ];then
    
    # Make filterbanks for the augmented data.  Note that we do not compute a new
    # vad.scp file here.  Instead, we use the vad.scp from the clean version of
    # the list.
    for name in voxceleb_aug_250k sitw_train_aug_3k
    do
	steps/make_mfcc.sh --mfcc-config conf/mfcc_16k.conf --nj 120 --cmd "$train_cmd" \
      			   data/$name exp/make_mfcc $mfccdir
	fix_data_dir.sh data/$name
    done

fi


if [ $stage -le 2 ];then
    # Combine the clean and augmented lists.  
    utils/combine_data.sh data/voxceleb_combined data/voxceleb_aug_250k data/voxceleb
    utils/combine_data.sh data/sitw_train_combined data/sitw_train_aug_3k data/sitw_train
  
fi
    
exit
