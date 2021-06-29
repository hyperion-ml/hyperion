#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $stage -le 1 ];then
  # Combine data to train x-vector nnet
    utils/combine_data.sh --extra-files "utt2dur utt2lang" \
			  data/swbd_sre_voxcelebcat_tel_proc_audio_no_sil \
			  data/swbd_proc_audio_no_sil data/sre_tel_proc_audio_no_sil data/voxcelebcat_tel_proc_audio_no_sil

    utils/combine_data.sh --extra-files "utt2dur utt2lang" \
			  data/sre16-8_proc_audio_no_sil \
			  data/sre16_train_dev_cmn_proc_audio_no_sil data/sre16_train_dev_ceb_proc_audio_no_sil \
			  data/sre16_eval_tr60_yue_proc_audio_no_sil data/sre16_eval_tr60_tgl_proc_audio_no_sil \
			  data/sre18_cmn2_train_lab_proc_audio_no_sil
    
    # utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    # 			  data/cvcat_noeng_tel_proc_audio_no_sil \
    # 			  $(echo $cv_noeng_datasets | sed -e 's@cvcat_@data/cvcat_@g' -e 's@_tel@_tel_proc_audio_no_sil@g')

    # utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    # 			  data/mls_tel_proc_audio_no_sil \
    # 			  $(echo $mls_datasets | sed -e 's@mls_@data/mls_@g' -e 's@_tel@_tel_proc_audio_no_sil@g')

    # utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    # 			  data/alleng_proc_audio_no_sil \
    # 			  data/swbd_sre_voxcelebcat_tel_proc_audio_no_sil \
    # 			  data/cvcat_en_tel_proc_audio_no_sil
    
    # utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    # 			  data/allnoeng_proc_audio_no_sil \
    # 			  data/sre16-8_proc_audio_no_sil \
    # 			  data/cvcat_noeng_tel_proc_audio_no_sil \
    # 			  data/cncelebcat_tel_proc_audio_no_sil \
    #                     data/fisher_spa_proc_audio_no_sil \
    #                     data/mls_tel_proc_audio_no_sil 
    
    # utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    # 			  data/alllangs_proc_audio_no_sil \
    # 			  data/alleng_proc_audio_no_sil \
    # 			  data/allnoeng_proc_audio_no_sil

    # utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    # 			  data/alllangs_nocv_proc_audio_no_sil \
    # 			  data/swbd_sre_voxcelebcat_tel_proc_audio_no_sil \
    # 			  data/sre16-8_proc_audio_no_sil \
    # 			  data/cncelebcat_tel_proc_audio_no_sil data/fisher_spa_proc_audio_no_sil

    # utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    # 			  data/alllangs_nocveng_proc_audio_no_sil \
    # 			  data/swbd_sre_voxcelebcat_tel_proc_audio_no_sil \
    # 			  data/sre16-8_proc_audio_no_sil \
    # 			  data/cncelebcat_tel_proc_audio_no_sil data/fisher_spa_proc_audio_no_sil \
    # 			  data/cvcat_noeng_tel_proc_audio_no_sil data/mls_tel_proc_audio_no_sil 

    utils/combine_data.sh --extra-files "utt2dur utt2lang" \
    			  data/alllangs_nocv_nocnceleb_proc_audio_no_sil \
    			  data/swbd_sre_voxcelebcat_tel_proc_audio_no_sil \
			  data/sre16-8_proc_audio_no_sil data/fisher_spa_proc_audio_no_sil

    utils/combine_data.sh --extra-files "utt2dur utt2lang" \
			  data/realtel_proc_audio_no_sil \
			  data/swbd_proc_audio_no_sil data/sre_tel_proc_audio_no_sil \
    			  data/sre16-8_proc_audio_no_sil \
			  data/fisher_spa_proc_audio_no_sil

    utils/combine_data.sh --extra-files "utt2dur utt2lang" \
			  data/realtelnoswbd_proc_audio_no_sil \
			  data/sre_tel_proc_audio_no_sil \
    			  data/sre16-8_proc_audio_no_sil \
			  data/fisher_spa_proc_audio_no_sil

    utils/combine_data.sh --extra-files "utt2dur utt2lang" \
			  data/realtelnoeng_proc_audio_no_sil \
			  data/sre16-8_proc_audio_no_sil \
			  data/fisher_spa_proc_audio_no_sil


fi

if [ $stage -le 2 ]; then
    # Prepare train and validation lists for x-vectors
    local/make_train_lists_sup_embed_with_augm.sh \
	data/${nnet_data}_proc_audio_no_sil \
	data/${nnet_data}_proc_audio_no_sil/lists_xvec
    exit
fi

if [ $stage -le 3 ]; then
    # Prepare adaptation lists for x-vectors
    if [ "$nnet_data" != "$nnet_adapt_data" ];then
	local/make_train_lists_sup_embed_with_augm.sh \
	    data/${nnet_adapt_data}_proc_audio_no_sil \
	    data/${nnet_adapt_data}_proc_audio_no_sil/lists_xvec
    fi
fi

exit
