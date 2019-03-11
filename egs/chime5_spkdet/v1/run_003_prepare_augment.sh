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
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
vaddiardir=`pwd`/vad_diar

stage=1

. parse_options.sh || exit 1;

# In this script, we augment the SWBD,SRE,MX6 and Voxceleb data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.

frame_shift=0.01

if [ $stage -le 1 ]; then

    if [ ! -d "RIRS_NOISES" ]; then
	if [ -d ../../sre18/v1.8k/RIRS_NOISES ];then
	    ln -s ../../sre18/v1.8k/RIRS_NOISES
	else
	    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
	    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
	    unzip rirs_noises.zip
	fi
    fi

    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    local/make_musan.sh /export/corpora/JHU/musan 16 data
    
    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in noise music; do
	utils/data/get_utt2dur.sh data/musan_${name}
	mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done
    
    steps_fe/get_utt2dur.sh --nj 20 --read_entire_file true data/mx6_mic
    cp data/mx6_mic/utt2dur data/mx6_mic/reco2dur 

fi

if [ $stage -le 2 ]; then
    
  for name in voxceleb sitw_train
  do
      awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/$name/utt2num_frames > data/$name/reco2dur
      
      # Make a reverberated version of the list.  Note that we don't add any
      # additive noise here.

      # Make a version with reverberated speech
      rvb_opts=()
      #rvb_opts+=(--rir-set-parameters "0.3, RIRS_NOISES/real_rirs_isotropic_noises/rir_list")
      rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
      rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
      
      python steps/data/reverberate_data_dir.py \
	      "${rvb_opts[@]}" \
	      --speech-rvb-probability 1 \
	      --pointsource-noise-addition-probability 0 \
	      --isotropic-noise-addition-probability 0 \
	      --num-replications 1 \
	      --source-sampling-rate 16000 \
	      data/${name} data/${name}_reverb
      cp data/${name}/vad.scp data/${name}_reverb/
      utils/copy_data_dir.sh --utt-suffix "-reverb" data/${name}_reverb data/${name}_reverb.new
      rm -rf data/${name}_reverb
      mv data/${name}_reverb.new data/${name}_reverb

      
      # Augment with musan_noise
      python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0:13:8" --fg-noise-dir "data/musan_noise" data/${name} data/${name}_noise
      # Augment with musan_music
      python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${name} data/${name}_music
      # Augment with mx6_mic
      python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13:10" --num-bg-noises "5:10:20:40" --bg-noise-dir "data/mx6_mic" data/${name} data/${name}_babble

      
      awk '{ $1=$1"-reverb"; print $0}' data/${name}/reco2dur > data/${name}_reverb/reco2dur
  
      # Augment with musan_noise
      python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0:13:8" --fg-noise-dir "data/musan_noise" data/${name}_reverb data/${name}_reverb_noise
      # Augment with musan_music
      python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${name}_reverb data/${name}_reverb_music
      # Augment with mx6_mic
      python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13:10" --num-bg-noises "5:10:20:40" --bg-noise-dir "data/mx6_mic" data/${name}_reverb data/${name}_reverb_babble


      # Combine noise only
      utils/combine_data.sh data/${name}_noise_all \
			    data/${name}_noise data/${name}_music data/${name}_babble

      # Combine reverbs
      utils/combine_data.sh data/${name}_reverb_all data/${name}_reverb \
			    data/${name}_reverb_noise data/${name}_reverb_music data/${name}_reverb_babble

      # keep halp of reverbs
      utils/subset_data_dir.sh data/${name}_reverb_all $(wc -l data/${name}_reverb_all/utt2spk | awk '{ print int($1/2)}') data/${name}_reverb_all_half

      
      # Combine reverb, noise, music, and babble into one directory.
      utils/combine_data.sh data/${name}_aug data/${name}_reverb_all_half data/${name}_noise_all

  done

fi

if [ $stage -le 3 ];then
    
  # Take a random subset of the augmentations 
  utils/subset_data_dir.sh data/voxceleb_aug 250000 data/voxceleb_aug_250k
  utils/fix_data_dir.sh data/voxceleb_aug_250k

  utils/subset_data_dir.sh data/sitw_train_aug 3000 data/sitw_train_aug_3k
  utils/fix_data_dir.sh data/sitw_train_aug_3k

fi
  
      
exit
