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

# We prepare the noise files and RIR for online speech augmentation
if [ $stage -le 1 ]; then

    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    local/make_musan.sh $musan_root 8 data
    
    for name in musan_noise musan_music
    do
	steps_xvec/preprocess_audios_for_nnet_train.sh --nj 10 --cmd "$train_cmd" \
	    --storage_name sre21-v1.8k-$(date +'%m_%d_%H_%M') \
	    data/${name} data/${name}_proc_audio exp/${name}_proc_audio
	utils/fix_data_dir.sh data/${name}_proc_audio
    done

fi

if [ $stage -le 2 ]; then

    # Create Babble noise from MUSAN speech files
    for name in musan_speech
    do
	steps_xvec/make_babble_noise_for_nnet_train.sh --cmd "$train_cmd" \
	    --storage_name sre21-v1.8k-$(date +'%m_%d_%H_%M') \
	    data/${name} data/${name}_babble exp/${name}_babble
	# utils/fix_data_dir.sh data/${name}_babble
    done
fi

if [ $stage -le 3 ]; then
    if [ ! -d "RIRS_NOISES" ]; then
	if [ -d ../v1.16k/RIRS_NOISES ];then
	    ln -s ../v1.16k/RIRS_NOISES
	else
	    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
	    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
	    unzip rirs_noises.zip
	fi
    fi
    local/make_rirs_data.sh RIRS_NOISES/simulated_rirs/smallroom 8 data/rirs_smallroom
    local/make_rirs_data.sh RIRS_NOISES/simulated_rirs/mediumroom 8 data/rirs_mediumroom
    local/make_rirs_data.sh RIRS_NOISES/real_rirs_isotropic_noises 8 data/rirs_real
    for rirs in rirs_smallroom rirs_mediumroom rirs_real
    do
	#pack all rirs in h5 files
	steps_xvec/pack_rirs_for_nnet_train.sh data/$rirs data/$rirs exp/rirs/$rirs
    done
    
fi


