#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
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

if [ "$feat_vers" == "kaldi" ];then
    make_fbank=steps/make_fbank.sh
    fbank_cfg=conf/fbank80_16k.conf
else
    fbank_cfg=conf/fbank80_16k.yaml
    if [ "$feat_vers" == "numpy" ];then
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
  for name in voxceleb2cat_train_augx${num_augs} 
  do
      $make_fbank --write-utt2num-frames true \
	  --fbank-config $fbank_cfg --nj 120 --cmd "$train_cmd" \
      	  data/$name exp/make_fbank/$name $fbankdir
      fix_data_dir.sh data/$name
  done

fi


if [ $stage -le 2 ];then
    
    # Combine the clean and augmented lists.  
    utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb2cat_train_combined data/voxceleb2cat_train_augx${num_augs} data/voxceleb2cat_train

fi
    
exit

