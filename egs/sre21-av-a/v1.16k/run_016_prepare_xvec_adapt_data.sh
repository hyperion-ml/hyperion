#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
nodes=c0
. parse_options.sh || exit 1;
. $config_file

export TMPDIR=data/tmp
mkdir -p $TMPDIR

if [ $stage -le 1 ];then
  # filter chinese speakers in sre and voxceleb
    for name in voxcelebcat \
		  voxcelebcat_8k
    do
      cp data/$name/utt2est_lang data/${name}_proc_audio_no_sil
      local/filter_chnspks_est_lang.sh \
	data/${name}_proc_audio_no_sil \
	data/${name}_chnspks_proc_audio_no_sil
    done

    for name in sre_cts_superset_16k_trn
    do
      local/filter_chnspks.sh \
	data/${name}_proc_audio_no_sil \
	data/${name}_chnspks_proc_audio_no_sil
    done
fi

if [ $stage -le 2 ];then
  utils/combine_data.sh \
    data/voxcelebcat_sre_alllangs_mixfs_chnspks_proc_audio_no_sil \
    data/voxcelebcat_chnspks_proc_audio_no_sil \
    data/voxcelebcat_8k_chnspks_proc_audio_no_sil \
    data/sre_cts_superset_16k_trn_chnspks_proc_audio_no_sil \
    data/sre16_eval_tr60_yue_proc_audio_no_sil \
    data/sre16_train_dev_cmn_proc_audio_no_sil
fi

if [ $stage -le 3 ]; then
    # Now, we remove files with less than 4s
    hyp_utils/remove_short_audios.sh --min-len 4 data/${nnet_adapt_data}_proc_audio_no_sil

    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 2 utterances.
    hyp_utils/remove_spk_few_utts.sh --min-num-utts 2 data/${nnet_adapt_data}_proc_audio_no_sil

fi

if [ $stage -le 4 ]; then
    # Prepare train and validation lists for x-vectors
    local/make_train_lists_sup_embed_with_augm.sh \
	data/${nnet_adapt_data}_proc_audio_no_sil \
	data/${nnet_adapt_data}_proc_audio_no_sil/lists_xvec

    # copy the class file from the original network so we dont need to retrain the output layer
    cp data/${nnet_data}_proc_audio_no_sil/lists_xvec/class2int \
       data/${nnet_adapt_data}_proc_audio_no_sil/lists_xvec
fi

exit
