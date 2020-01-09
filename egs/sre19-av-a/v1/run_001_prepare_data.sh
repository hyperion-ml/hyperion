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

# I will just use voxceleb so I start from stage 2
stage=2

config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 


# if [ $stage -le 1 ]; then
#     # Path to some, but not all of the training corpora


#     if [ ! -f $master_key ];then
# 	# Get the sre04-12 master key
# 	# Try link to telephone part
# 	if [ -d ../../sre19-cmn2/v1/$master_key_dir ];then
# 	    ln -s ../../sre19-cmn2/v1/$master_key_dir
# 	else
# 	    # Download from google drive.
# 	    local/download_sre04-12_master_key.sh
# 	fi
#     fi
    
#     # Prepare telephone and microphone speech from Mixer6.
#     local/make_mx6.sh $ldc_root/LDC2013S03 16 data/
   
#     # Prepare sre10 interview and mic phonecalls 
#     local/make_sre10mic.sh $sre10_16k_root $master_key 16 data

#     # Prepare sre12
#     local/make_sre12.sh $sre12_root $master_key 16 data/

#     # Combine all SRE+MX6 mic phonecalls into one dataset
#     utils/combine_data.sh --extra-files utt2info data/sre_phnmic \
#     			  data/sre10_phnmic data/sre12_phnmic data/mx6_mic
#     utils/validate_data_dir.sh --no-text --no-feats data/sre_phnmic

# fi


if [ $stage -le 2 ];then
    # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
    # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
    # contains the speakers that overlap between VoxCeleb1 and our evaluation
    # set SITW.  The script removes these overlapping speakers from VoxCeleb1.
    local/make_voxceleb1cat.pl $voxceleb1_root 16 data

    # Prepare the dev portion of the VoxCeleb2 dataset.
    local/make_voxceleb2cat.pl $voxceleb2_root dev 16 data/voxceleb2cat_train

fi


if [ $stage -le 3 ];then
    # Prepare SITW dev to train x-vector
    # local/make_sitw_train.sh $sitw_root dev 16 data/sitw_train_dev
    
    # Make SITW dev and eval sets
    local/make_sitw.sh $sitw_root 16 data/sitw
fi

if [ $stage -le 4 ];then
    # Prepare sre18
    local/make_sre18_dev.sh $sre18_dev_root 16 data
    local/make_sre18_eval.sh $sre18_eval_root 16 data
fi

if [ $stage -le 5 ];then
    # Prepare sre19
    local/make_sre19av_a_dev.sh $sre19_dev_root 16 data
    local/make_sre19av_a_eval.sh $sre19_eval_root 16 data
fi

if [ $stage -le 6 ];then
    local/make_janus_core.sh $janus_root 16 data
fi

if [ $stage -le 7 ];then
    local/make_dihard_train.sh $dihard2_dev dev data
    local/make_dihard_train.sh $dihard2_eval eval data
fi

exit
