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
    
  # Make MFCC for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  for name in swbd_sre_tel_augx${num_augs} voxcelebcat_tel_augx${num_augs}  sre18_cmn2_adapt_lab_augx${num_augs}
  do
      steps/make_mfcc.sh --write-utt2num-frames true \
	  --mfcc-config conf/mfcc_8k.conf --nj 120 --cmd "$train_cmd" \
      	  data/$name exp/make_mfcc/$name $mfccdir
      fix_data_dir.sh data/$name
  done

fi


if [ $stage -le 2 ];then

    # Combine the clean and augmented lists.

    utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre_tel_combined data/swbd_sre_tel_augx${num_augs} data/swbd_sre_tel
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxcelebcat_tel_combined data/voxcelebcat_tel_augx${num_augs} data/voxcelebcat_tel
    utils/combine_data.sh --extra-files "utt2num_frames" data/sre18_cmn2_adapt_lab_combined data/sre18_cmn2_adapt_lab_augx${num_augs} data/sre18_cmn2_adapt_lab

fi

if [ $stage -le 3 ];then
    # Filter out the clean + augmented portion of the SRE list.
    utils/copy_data_dir.sh data/swbd_sre_tel_combined data/sre_tel_combined
    utils/filter_scp.pl data/sre_tel/spk2utt data/swbd_sre_tel_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_tel_combined/utt2spk
    utils/fix_data_dir.sh data/sre_tel_combined
fi
exit

if [ $stage -le 2 ];then
    
    # Combine the clean and augmented lists.  
    utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre_tel_combined data/swbd_sre_tel_augx${num_augs} data/swbd_sre_tel
    #utils/combine_data.sh --extra-files "utt2num_frames" data/sre_phnmic_combined data/sre_phnmic_augx5 data/sre_phnmic
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb_combined data/voxceleb_augx${num_augs} data/voxceleb
    utils/combine_data.sh --extra-files "utt2num_frames" data/sre18_train_eval_cmn2_combined data/sre18_train_eval_cmn2_augx${num_augs} data/sre18_train_eval_cmn2
    #utils/combine_data.sh --extra-files "utt2num_frames" data/mgb2_train_mer80_combined data/mgb2_train_mer80_augx${num_augs} data/mgb2_train_mer80

    # Filter out the clean + augmented portion of the SRE list.  
    utils/copy_data_dir.sh data/swbd_sre_tel_combined data/sre_tel_combined
    utils/filter_scp.pl data/sre_tel/spk2utt data/swbd_sre_tel_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_tel_combined/utt2spk
    utils/fix_data_dir.sh data/sre_tel_combined
  
fi
    
exit
