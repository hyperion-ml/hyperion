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
    fbank_cfg=conf/fbank64_8k.conf
else
    fbank_cfg=conf/fbank64_8k.yaml
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
  for name in swbd_sre_tel_augx${num_augs} voxcelebcat_tel_augx${num_augs} sre18_cmn2_adapt_lab_augx${num_augs}
  do
      $make_fbank --write-utt2num-frames true \
	  --fbank-config $fbank_cfg --nj 120 --cmd "$train_cmd --mem 16G" \
      	  data/$name exp/make_fbank/$name $fbankdir
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

