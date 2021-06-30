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

if [ $stage -le 1 ]; then
    # This script preprocess audio for x-vector training
    for name in sre_tel swbd voxcelebcat_tel \
    			sre16_train_dev_cmn sre16_train_dev_ceb \
    			sre16_eval_tr60_yue sre16_eval_tr60_tgl \
    		        sre18_cmn2_train_lab fisher_spa \
    			# cncelebcat_tel \
    			# $cv_datasets $mls_datasets
    do
	steps_xvec/preprocess_audios_for_nnet_train.sh --nj 40 --cmd "$train_cmd" \
	    --storage_name sre20-cts-v1-$name-$(date +'%m_%d_%H_%M') --use-bin-vad true \
	    data/${name} data/${name}_proc_audio_no_sil exp/${name}_proc_audio_no_sil
	utils/fix_data_dir.sh data/${name}_proc_audio_no_sil
    done
fi

if [ $stage -le 2 ]; then
    for name in sre_tel swbd voxcelebcat_tel \
    			sre16_train_dev_cmn sre16_train_dev_ceb \
    			sre16_eval_tr60_yue sre16_eval_tr60_tgl \
    		        sre18_cmn2_train_lab \
    			cncelebcat_tel fisher_spa \
    			$cv_datasets 
    do
	# Now, we remove files with less than 4s
	hyp_utils/remove_short_audios.sh --min-len 4 data/${name}_proc_audio_no_sil
	
	# We also want several utterances per speaker. Now we'll throw out speakers
	# with fewer than 4 utterances.
	hyp_utils/remove_spk_few_utts.sh --min-num-utts 4 data/${name}_proc_audio_no_sil
    done

    # for name in $mls_datasets
    # do
    # 	# Now, we remove files with less than 4s
    # 	hyp_utils/remove_short_audios.sh --min-len 4 data/${name}_proc_audio_no_sil
	
    # 	# We also want several utterances per speaker. Now we'll throw out speakers
    # 	# with fewer than 4 utterances.
    # 	hyp_utils/remove_spk_few_utts.sh --min-num-utts 2 data/${name}_proc_audio_no_sil
    # done

fi


exit
