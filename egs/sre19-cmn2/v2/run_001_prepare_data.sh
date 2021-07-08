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

. parse_options.sh || exit 1;
. datapath.sh 

if [ $stage -le 1 ]; then
    # Path to some, but not all of the training corpora

    if [ ! -f $master_key ];then
	# Get the sre04-12 master key
	# Try link to v2
	if [ -d ../v2/$master_key_dir ];then
	    ln -s ../v2/$master_key_dir
	else
	    # Download from google drive.
	    local/download_sre04-12_master_key.sh
	fi
    fi
    
    # Prepare telephone and microphone speech from Mixer6.
    local/make_mx6.sh $ldc_root/LDC2013S03 8 data/
   
    # Prepare sre04-06 telephone speech
    local/make_sre04-06.sh $ldc_root $master_key 8 data

    # Prepare sre08
    local/make_sre08.sh $ldc_root $master_key 8 data
    
    # Prepare sre08 supplemental
    # local/make_sre08sup.sh $sre08sup_root $master_key 8 data

    # Prepare sre10 tel 
    local/make_sre10tel.sh $sre10_root $master_key 8 data

    # Prepare sre10 interview and mic phonecalls 
    # local/make_sre10mic.sh $sre10_16k_root $master_key 8 data

    # Prepare sre12
    local/make_sre12.sh $sre12_root $master_key 8 data/

    # Combine all SRE+MX6 tel into one dataset
    utils/combine_data.sh --extra-files utt2info data/sre_tel \
    			  data/sre04-06 data/sre08_tel data/sre10_tel data/sre12_tel data/mx6_calls
    utils/validate_data_dir.sh --no-text --no-feats data/sre_tel
    
    # Combine all SRE+MX6 mic phonecalls into one dataset
    # utils/combine_data.sh --extra-files utt2info data/sre_phnmic \
    # 			  data/sre08_phnmic data/sre10_phnmic data/sre12_phnmic data/mx6_mic
    # utils/validate_data_dir.sh --no-text --no-feats data/sre_phnmic

fi


if [ $stage -le 2 ];then

    # Prepare SWBD corpora.
    local/make_swbd_cellular1.pl $ldc_root/LDC2001S13 \
    				 data/swbd_cellular1_train
    local/make_swbd_cellular2.pl $swbd_cell2_root \
    				 data/swbd_cellular2_train
    local/make_swbd2_phase1.pl $swbd2_ph1_root \
			       data/swbd2_phase1_train
    local/make_swbd2_phase2.pl $ldc_root/LDC99S79 \
			       data/swbd2_phase2_train
    local/make_swbd2_phase3.pl $ldc_root/LDC2002S06 \
			       data/swbd2_phase3_train
    
    # Combine all SWB corpora into one dataset.
    utils/combine_data.sh data/swbd \
			  data/swbd_cellular1_train data/swbd_cellular2_train \
			  data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

fi


if [ $stage -le 3 ];then
    # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
    # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
    # contains the speakers that overlap between VoxCeleb1 and our evaluation
    # set SITW.  The script removes these overlapping speakers from VoxCeleb1.
    local/make_voxceleb1cat.pl $voxceleb1_root 8 data

    # Prepare the dev portion of the VoxCeleb2 dataset.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 8 data/voxceleb2cat_train
    local/make_voxceleb2cat.pl $voxceleb2_root test 8 data/voxceleb2cat_test

    utils/combine_data.sh data/voxcelebcat data/voxceleb1cat data/voxceleb2cat_train
    local/apply_sox_tel_codec.sh data/voxcelebcat data/voxcelebcat_tel

fi


if [ $stage -le 4 ];then
    # Prepare sre18 for train and eval labeled
    local/make_sre18_train_dev.sh $sre18_dev_root 8 data
    local/make_sre18_eval_tr60_ev40.sh $sre18_eval_root 8 data
    utils/combine_data.sh data/sre18_cmn2_adapt_lab data/sre18_train_dev_cmn2 data/sre18_eval_cmn2_tr60
fi


if [ $stage -le 5 ];then
    # Prepare sre18 unlabeled
    local/make_sre18_dev.sh $sre18_dev_root 8 data
fi

if [ $stage -le 6 ];then
    # Prepare sre19
    local/make_sre19cmn2_eval.sh $sre19cmn2_eval_root 8 data
fi



