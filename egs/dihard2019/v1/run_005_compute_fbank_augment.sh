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
fbankdir=`pwd`/exp/fbank

stage=1
config_file=default_config.sh
feat_vers="numpy"

. parse_options.sh || exit 1;
. $config_file

fbank_cfg=conf/fbank80_16k.pyconf
if [ "$feat_vers" == "numpy" ];then
    make_fbank=hyp_utils/feats/make_fbank.sh
else
    make_fbank=hyp_utils/feats/make_torch_fbank.sh
fi


export TMPDIR=data/tmp
mkdir -p $TMPDIR

if [ $stage -le 1 ];then
    
  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  for name in voxceleb1cat_train_augx${num_augs} voxceleb2cat_augx${num_augs} 
  do
      $make_fbank --write-utt2num-frames true \
	  --fbank-config $fbank_cfg --nj 120 --cmd "$train_cmd" \
      	  data/$name exp/make_fbank/$name $fbankdir
      fix_data_dir.sh data/$name
  done

fi


if [ $stage -le 2 ];then
    
    # Combine the clean and augmented lists.  
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb1cat_combined data/voxceleb1cat_train_augx${num_augs} data/voxceleb1cat_train
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb2cat_combined data/voxceleb2cat_augx${num_augs} data/voxceleb2cat

fi
    
if [ $stage -le 3 ];then
    # Combine Voxceleb 1+2
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxcelebcat_combined data/voxceleb1cat_combined data/voxceleb2cat_combined
fi

exit

