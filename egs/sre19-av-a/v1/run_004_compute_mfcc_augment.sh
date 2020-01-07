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
mfccdir=`pwd`/exp/mfcc

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

export TMPDIR=data/tmp
mkdir -p $TMPDIR

if [ $stage -le 1 ];then
    
  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  for name in voxceleb_augx${num_augs} #dihard2_train_augx${num_augs} 
  do
      steps/make_mfcc.sh --write-utt2num-frames true \
	  --mfcc-config conf/mfcc_16k.conf --nj 120 --cmd "$train_cmd" \
      	  data/$name exp/make_mfcc/$name $mfccdir
      fix_data_dir.sh data/$name
  done

fi


if [ $stage -le 2 ];then
    # Combine the clean and augmented lists.  
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb_combined data/voxceleb_augx${num_augs} data/voxceleb
    #utils/combine_data.sh --extra-files "utt2num_frames" data/dihard2_train_combined data/dihard2_train_augx${num_augs} data/dihard2_train
fi
    
exit

