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
feat_vers="kaldi"

. parse_options.sh || exit 1;
. $config_file

if [ "feat_vers" == "kaldi" ];then
    make_fbank=steps/make_fbank.sh
    fbank_cfg=conf/fbank80_16k.conf
else
    fbank_cfg=conf/fbank80_16k.pyconf
    if [ "feat_vers" == "numpy" ];then
	make_fbank=steps_pyfe/make_fbank.sh
    else
	make_fbank=steps_pyfe/make_torch_fbank.sh
    fi
fi

export TMPDIR=data/tmp
mkdir -p $TMPDIR

if [ $stage -le 1 ];then
    
  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  for name in voxcelebcat_augx${num_augs} #dihard2_train_augx${num_augs} 
  do
      $make_fbank --write-utt2num-frames true \
	  --fbank-config $fbank_cfg --nj 120 --cmd "$train_cmd" \
      	  data/$name exp/make_fbank/$name $fbankdir
      fix_data_dir.sh data/$name
  done

fi


if [ $stage -le 2 ];then
    
    # Combine the clean and augmented lists.  
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxcelebcat_combined data/voxcelebcat_augx${num_augs} data/voxcelebcat
    # utils/combine_data.sh --extra-files "utt2num_frames" data/dihard2_train_combined data/dihard2_train_augx${num_augs} data/dihard2_train

fi
    
exit

