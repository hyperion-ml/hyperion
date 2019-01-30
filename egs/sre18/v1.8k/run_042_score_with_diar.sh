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

# SITW Trials
sitw_dev_trials_core=data/sitw_dev_test/trials/core-core.lst
sitw_eval_trials_core=data/sitw_eval_test/trials/core-core.lst
sitw_dev_trials_core_multi=data/sitw_dev_test/trials/core-multi.lst
sitw_eval_trials_core_multi=data/sitw_eval_test/trials/core-multi.lst
sitw_dev_trials_assist_core=data/sitw_dev_test/trials/assist-core.lst
sitw_eval_trials_assist_core=data/sitw_eval_test/trials/assist-core.lst
sitw_dev_trials_assist_multi=data/sitw_dev_test/trials/assist-multi.lst
sitw_eval_trials_assist_multi=data/sitw_eval_test/trials/assist-multi.lst


# SRE16 trials
sre16_trials_dev=data/sre16_dev_test/trials
sre16_trials_eval=data/sre16_eval_test/trials
sre16_trials_ceb=data/sre16_dev_test/trials_ceb
sre16_trials_cmn=data/sre16_dev_test/trials_cmn
sre16_trials_tgl=data/sre16_eval_test/trials_tgl
sre16_trials_yue=data/sre16_eval_test/trials_yue

# SRE18 trials
sre18_dev_trials_cmn2=data/sre18_dev_test_cmn2/trials
sre18_dev_trials_cmn2_pstn=data/sre18_dev_test_cmn2/trials_pstn
sre18_dev_trials_cmn2_pstn_samephn=data/sre18_dev_test_cmn2/trials_pstn_samephn
sre18_dev_trials_cmn2_pstn_diffphn=data/sre18_dev_test_cmn2/trials_pstn_diffphn
sre18_dev_trials_cmn2_voip=data/sre18_dev_test_cmn2/trials_voip
sre18_dev_trials_vast=data/sre18_dev_test_vast/trials
sre18_eval_trials_cmn2=data/sre18_eval_test_cmn2/trials
sre18_eval_trials_vast=data/sre18_eval_test_vast/trials

diar_thr=-0.9
min_dur=10
rttm_dir=../diarization.v1.16k/exp/scores/1b16k/lda120_plda_voxceleb
diar_name=diar1b

#Diarization Ndx
sitw_eval_trials_core_diar=data/sitw_eval_test_${diar_name}/trials/core-core.lst
sitw_eval_trials_core_multi_diar=data/sitw_eval_test_${diar_name}/trials/core-multi.lst
sitw_eval_trials_assist_core_diar=data/sitw_eval_test_${diar_name}/trials/assist-core.lst
sitw_eval_trials_assist_multi_diar=data/sitw_eval_test_${diar_name}/trials/assist-multi.lst
sre18_dev_trials_vast_diar=data/sre18_dev_test_vast_${diar_name}/trials
sre18_eval_trials_vast_diar=data/sre18_eval_test_vast_${diar_name}/trials

net_name=1a
nnet_dir=exp/xvector_nnet_$net_name
xvector_dir=exp/xvectors/$net_name

tel_lda_dim=150
vid_lda_dim=200
#cmn2_lda_dim=50
tel_ncoh=400
vid_ncoh=500
vast_ncoh=120

w_mu1=1
w_B1=0.75
w_W1=0.75
w_mu2=1
w_B2=0.6
w_W2=0.6
num_spks=975
# w_mu3=1
# w_B3=0.1
# w_W3=0.15
# w_mu4=1
# w_B4=0.1
# w_W4=0.15

plda_tel_y_dim=125
plda_tel_z_dim=150
plda_vid_y_dim=150
plda_vid_z_dim=200

stage=21

. parse_options.sh || exit 1;

# coh_vid_data=sitw_dev
coh_vid_data=sitw_sre18_dev_vast_${diar_name}
coh_vast_data=sitw_sre18_dev_vast_${diar_name}
coh_tel_data=sre18_dev_unlabeled
plda_tel_data=sretel_combined
# plda_tel_type=frplda
# plda_tel_label=${plda_tel_type}_adaptmu${w_mu}B${w_B}W${w_W}
# plda_tel_label=${plda_tel_type}_adapt1_mu${w_mu1}B${w_B1}W${w_W1}_adapt2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}
# plda_tel_label=${plda_tel_type}_adapt1_mu${w_mu1}B${w_B1}W${w_W1}_adapt2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}_adapt3_mu${w_mu3}B${w_B3}W${w_W3}
# plda_tel_label=${plda_tel_type}_adapt1_mu${w_mu1}B${w_B1}W${w_W1}_adapt2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}_adapt3_mu${w_mu3}B${w_B3}W${w_W3}_adapt4_mu${w_mu4}B${w_B4}W${w_W4}
plda_tel_type=splda
plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_v2_adapt1_mu${w_mu1}B${w_B1}W${w_W1}_adapt2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}
#plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt1_mu${w_mu1}B${w_B1}W${w_W1}_adapt2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}_adapt3_mu${w_mu3}B${w_B3}W${w_W3}_adapt4_mu${w_mu4}B${w_B4}W${w_W4}
#plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt1_mu${w_mu1}B${w_B1}W${w_W1}_adapt2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}_adapt3_lda${cmn2_lda_dim}_mu${w_mu3}B${w_B3}W${w_W3}
#plda_tel_type=plda
#plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}z${plda_tel_z_dim}_adapt1_mu${w_mu1}B${w_B1}W${w_W1}_adapt2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}

plda_vid_data=vcsitwtrn_combined
# plda_vid_type=frplda
# plda_vid_label=${plda_vid_type}
plda_vid_type=splda
plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v3
# plda_vid_type=plda
# plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}z${plda_vid_z_dim}

be_tel_name=lda${tel_lda_dim}_${plda_tel_label}_${plda_tel_data}
be_vid_name=lda${vid_lda_dim}_${plda_vid_label}_${plda_vid_data}
be_tel_dir=exp/be/$net_name/$be_tel_name
be_vid_dir=exp/be/$net_name/$be_vid_name

score_dir=exp/scores/$net_name/tel_${be_tel_name}_vid_${be_vid_name}
score_plda_out_dir=$score_dir/plda_out

scoring_software_dir=./scoring_software/
scoring_software_sre18_dir=./scoring_software_sre18/
master_key=/export/b17/janto/SRE18/master_key/key/NIST_SRE_segments_key.csv

data_root=/export/corpora/LDC
sitw_root=/export/corpora/SRI/sitw
sre12_root=/export/b17/janto/SRE18/corpora/LDC2016E45
voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
sre18_dev_root=/export/corpora/LDC/LDC2018E46
sre18_eval_root=/export/b17/janto/SRE18/corpora/LDC2018E51
sre18_dev_meta=${sre18_dev_root}/docs/sre18_dev_segment_key.tsv


if [ $stage -le 0 ]; then
    # Path to some, but not all of the training corpora
    
  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $data_root/LDC2013S03 data/

  # Prepare SRE12
  local/make_sre12.sh $sre12_root $master_key data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/
  
  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh $data_root data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1 data/sre2006_test_2 \
    data/sre08 data/mx6 data/sre10 data/sre12_tel_phn data/sre12_mic_phn
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare NIST SRE 2016 evaluation data.
  local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data
  local/make_sre16_dev.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data

  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data

  # Prepare SITW dev to train x-vector
  local/make_sitw_train.sh $sitw_root dev 8 data/sitw_train_dev
  
  # Make SITW dev and eval sets
  local/make_sitw.sh $sitw_root data/sitw

  # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
  # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
  # contains the speakers that overlap between VoxCeleb1 and our evaluation
  # set SITW.  The script removes these overlapping speakers from VoxCeleb1.
  local/make_voxceleb1cat.pl $voxceleb1_root 8 data

  # Prepare the dev portion of the VoxCeleb2 dataset.
  local/make_voxceleb2cat.pl $voxceleb2_root dev 8 data/voxceleb2cat_train

  # Prepare sre18
  local/make_sre18_dev_8k.sh $sre18_dev_root data
  local/make_sre18_eval_8k.sh $sre18_eval_root data
  exit
fi


if [ $stage -le 1 ]; then
  # Make filterbanks and compute the energy-based VAD for each dataset
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b{14,15,16,17}/$USER/kaldi-data/egs/sre18/v1/xvector-$(date +'%m_%d_%H_%M')/mfccs/storage $mfccdir/storage
  fi
  for name in sre swbd sre16_eval_enroll sre16_eval_test sre16_dev_enroll sre16_dev_test sre16_major sre16_minor \
  		  sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test sitw_train_dev\
  		  voxceleb1cat voxceleb2cat_train \
  		  sre18_dev_unlabeled sre18_dev_enroll_cmn2 sre18_dev_test_cmn2 \
		  sre18_eval_enroll_cmn2 sre18_eval_test_cmn2 sre18_eval_test_vast;
  do
      steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
			 data/${name} exp/make_mfcc $mfccdir
      utils/fix_data_dir.sh data/${name}
      sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
				  data/${name} exp/make_vad $vaddir
      utils/fix_data_dir.sh data/${name}
  done

  for name in sre18_dev_test_vast; do
      steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 5 --cmd "$train_cmd" \
    			 data/${name} exp/make_mfcc $mfccdir
      utils/fix_data_dir.sh data/${name}
      sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
    				  data/${name} exp/make_vad $vaddir
      utils/fix_data_dir.sh data/${name}
  done
  
  
  for name in sre18_dev_enroll_vast; do
      steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
    			 data/${name} exp/make_mfcc $mfccdir
      utils/fix_data_dir.sh data/${name}
      local/sre18_diar_to_vad.sh data/${name} exp/make_vad $vaddir
      utils/fix_data_dir.sh data/${name}
  done

  for name in sre18_eval_enroll_vast; do
      steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
    			 data/${name} exp/make_mfcc $mfccdir
      utils/fix_data_dir.sh data/${name}
      local/sre18_diar_to_vad.sh data/${name} exp/make_vad $vaddir
      utils/fix_data_dir.sh data/${name}
  done

  
  utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre data/swbd data/sre
  utils/fix_data_dir.sh data/swbd_sre
  utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb data/voxceleb1cat data/voxceleb2cat_train
  utils/fix_data_dir.sh data/voxceleb
  exit
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01


  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /export/corpora/JHU/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done


  for name in swbd_sre voxceleb sitw_train_dev
  do
      awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/$name/utt2num_frames > data/$name/reco2dur
      
      # Make a reverberated version of the list.  Note that we don't add any
      # additive noise here.
      python2 steps/data/reverberate_data_dir.py \
	      "${rvb_opts[@]}" \
	      --speech-rvb-probability 1 \
	      --pointsource-noise-addition-probability 0 \
	      --isotropic-noise-addition-probability 0 \
	      --num-replications 1 \
	      --source-sampling-rate 8000 \
	      data/${name} data/${name}_reverb
      cp data/${name}/vad.scp data/${name}_reverb/
      utils/copy_data_dir.sh --utt-suffix "-reverb" data/${name}_reverb data/${name}_reverb.new
      rm -rf data/${name}_reverb
      mv data/${name}_reverb.new data/${name}_reverb

      
      # Augment with musan_noise
      python2 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${name} data/${name}_noise
      # Augment with musan_music
      python2 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${name} data/${name}_music
      # Augment with musan_speech
      python2 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${name} data/${name}_babble

      
      awk '{ $1=$1"-reverb"; print $0}' data/${name}/reco2dur > data/${name}_reverb/reco2dur
  
      # Augment with musan_noise
      python2 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${name}_reverb data/${name}_reverb_noise
      # Augment with musan_music
      python2 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${name} data/${name}_reverb_music
      # Augment with musan_speech
      python2 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${name} data/${name}_reverb_babble

  
      # Combine reverb, noise, music, and babble into one directory.
      utils/combine_data.sh data/${name}_aug data/${name}_reverb data/${name}_noise data/${name}_music data/${name}_babble \
			    data/${name}_reverb_noise data/${name}_reverb_music data/${name}_reverb_babble

  done
  exit
fi

if [ $stage -le 3 ];then
    
  # Take a random subset of the augmentations 
  utils/subset_data_dir.sh data/swbd_sre_aug 200000 data/swbd_sre_aug_200k
  utils/fix_data_dir.sh data/swbd_sre_aug_200k

  utils/subset_data_dir.sh data/voxceleb_aug 250000 data/voxceleb_aug_250k
  utils/fix_data_dir.sh data/voxceleb_aug_250k

  utils/subset_data_dir.sh data/sitw_train_dev_aug 1200 data/sitw_train_dev_aug_1200
  utils/fix_data_dir.sh data/sitw_train_dev_aug_1200

  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  for name in swbd_sre_aug_200k voxceleb_aug_250k sitw_train_dev_aug_1200
  do
      steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 120 --cmd "$train_cmd" \
      			 data/$name exp/make_mfcc $mfccdir
      fix_data_dir.sh data/$name
  done

  # Combine the clean and augmented lists.  
  utils/combine_data.sh data/swbd_sre_combined data/swbd_sre_aug_200k data/swbd_sre
  utils/combine_data.sh data/voxceleb_combined data/voxceleb_aug_250k data/voxceleb
  utils/combine_data.sh data/sitw_train_combined data/sitw_train_dev_aug_1200 data/sitw_train_dev

  # Filter out the clean + augmented portion of the SRE list.  
  utils/copy_data_dir.sh data/swbd_sre_combined data/sre_combined
  utils/filter_scp.pl data/sre/spk2utt data/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
  utils/fix_data_dir.sh data/sre_combined

  # Combine data to train x-vector nnet
  utils/combine_data.sh data/train_combined data/swbd_sre_combined data/voxceleb_combined data/sitw_train_combined
  exit
fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd -l \"hostname=b[01]*\" -V" \
    data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
  utils/fix_data_dir.sh data/train_combined_no_sil
  exit
fi


if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 4s (400 frames) per utterance.
  min_len=400
  mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
  mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
  mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk

  utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
  mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_combined_no_sil
  exit
fi

if [ $stage -le 8 ]; then
    local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 --num_epochs 2 \
				       --data data/train_combined_no_sil --nnet-dir $nnet_dir \
				       --egs-dir $nnet_dir/egs
fi



if [ $stage -le 9 ]; then
    # Extract xvectors for SRE(includes Mixer 6)/Voxceleb/SITW data . We'll use this for
    # things like LDA or PLDA.

    for name in sre voxceleb sitw_train
    do
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 60 \
					      $nnet_dir data/${name}_combined \
					      $xvector_dir/${name}_combined
    done

    # Extracts x-vectors for evaluation
    for name in sre16_eval_enroll sre16_eval_test sre16_dev_enroll sre16_dev_test sre16_major sre16_minor \
  				  sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test \
  				  sre18_dev_unlabeled sre18_dev_enroll_cmn2 sre18_dev_test_cmn2 \
				  sre18_eval_enroll_cmn2 sre18_eval_test_cmn2 \
				  sre18_eval_enroll_vast sre18_eval_test_vast

    do
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done

    for name in sre18_dev_enroll_vast sre18_dev_test_vast
    do
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 1 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done

    mkdir -p $xvector_dir/train_combined
    cat $xvector_dir/{sre,voxceleb,sitw_train}_combined/xvector.scp > $xvector_dir/train_combined/xvector.scp

    utils/combine_data.sh --skip_fix true data/vcsitwtrn_combined data/voxceleb_combined data/sitw_train_combined
    local/fix_data_dir2.sh data/vcsitwtrn_combined
    local/validate_data_dir2.sh --no-text --no-spk-sort data/vcsitwtrn_combined
    
    mkdir -p $xvector_dir/vcsitwtrn_combined
    cat $xvector_dir/{voxceleb,sitw_train}_combined/xvector.scp > $xvector_dir/vcsitwtrn_combined/xvector.scp

    # mkdir -p data/mx6_combined
    # awk '/_MX6_/ { print $1}' data/train_combined/utt2spk > data/mx6_combined/utts
    # utils/subset_data_dir.sh --utt-list data/mx6_combined/utts data/train_combined data/mx6_combined
    
    # utils/combine_data.sh --skip_fix true data/vcsitwtrnmx6_combined data/voxceleb_combined data/sitw_train_combined data/mx6_combined
    # local/fix_data_dir2.sh data/vcsitwtrnmx6_combined
    # local/validate_data_dir2.sh --no-text --no-spk-sort data/vcsitwtrnmx6_combined

    # mkdir -p $xvector_dir/vcsitwtrnmx6_combined
    # utils/filter_scp.pl data/vcsitwtrnmx6_combined/utt2spk $xvector_dir/train_combined/xvector.scp > $xvector_dir/vcsitwtrnmx6_combined/xvector.scp

    local/filter_tel.sh data/sre_combined $master_key data/sretel_combined
    mkdir -p $xvector_dir/sretel_combined
    utils/filter_scp.pl data/sretel_combined/utt2spk $xvector_dir/sre_combined/xvector.scp > $xvector_dir/sretel_combined/xvector.scp
    
    exit
fi

if [ $stage -le 10 ]; then
    # Create datasets with diarization
    for name in sitw_dev_test sitw_eval_test sre18_dev_test_vast sre18_eval_test_vast
    do
	rttm=$rttm_dir/$name/plda_scores_t${diar_thr}/rttm
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur data/$name $rttm data/${name}_${diar_name} $vaddiardir
    done

    mkdir -p data/sitw_eval_test_${diar_name}/trials
    for file in assist-core.lst assist-multi.lst core-core.lst core-multi.lst
    do
	local/make_diar_trials.sh data/sitw_eval_test_${diar_name}/orig2utt data/sitw_eval_test/trials/$file data/sitw_eval_test_${diar_name}/trials/$file 
    done
    local/make_diar_trials.sh data/sre18_dev_test_vast_${diar_name}/orig2utt data/sre18_dev_test_vast/trials data/sre18_dev_test_vast_${diar_name}/trials
    local/make_diar_trials.sh data/sre18_eval_test_vast_${diar_name}/orig2utt data/sre18_eval_test_vast/trials data/sre18_eval_test_vast_${diar_name}/trials
    exit
fi


if [ $stage -le 11 ]; then
    # Extract xvectors for diarization data

    for name in sitw_dev_test_${diar_name} sitw_eval_test_${diar_name} sre18_eval_test_vast_${diar_name}
    do
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done

    for name in sre18_dev_test_vast_${diar_name}
    do
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 1 \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done

    exit
fi

# for name in sitw_dev_test 
# do
#     rttm=$rttm_dir/$name/plda_scores_t${diar_thr}/rttm
#     local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur data/$name $rttm data/${name}_${diar_name} $vaddiardir
# done

    # for name in sitw_dev_test_${diar_name} 
    # do
    # 	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    # 					      $nnet_dir data/$name \
    # 					      $xvector_dir/$name
    # done
#exit


#export train_cmd="run.pl"

if [ $stage -le 20 ]; then
    
    utils/combine_data.sh data/sitw_dev data/sitw_dev_enroll data/sitw_dev_test
    mkdir -p $xvector_dir/sitw_dev
    cat $xvector_dir/sitw_dev_{enroll,test}/xvector.scp > $xvector_dir/sitw_dev/xvector.scp

    mkdir -p $xvector_dir/sitw_eval
    cat $xvector_dir/sitw_eval_{enroll,test}/xvector.scp > $xvector_dir/sitw_eval/xvector.scp


    utils/combine_data.sh data/sitw_dev_${diar_name} data/sitw_train_combined data/sitw_dev_test_${diar_name}
    mkdir -p $xvector_dir/sitw_dev_${diar_name}
    cat $xvector_dir/sitw_{train_combined,dev_test_${diar_name}}/xvector.scp > $xvector_dir/sitw_dev_${diar_name}/xvector.scp

    utils/combine_data.sh data/sre18_dev_vast data/sre18_dev_enroll_vast data/sre18_dev_test_vast 
    mkdir -p $xvector_dir/sre18_dev_vast
    cat $xvector_dir/sre18_dev_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_dev_vast/xvector.scp

    utils/combine_data.sh data/sre18_dev_vast_${diar_name} data/sre18_dev_enroll_vast data/sre18_dev_test_vast_${diar_name} 
    mkdir -p $xvector_dir/sre18_dev_vast_${diar_name}
    cat $xvector_dir/sre18_dev_{enroll_vast,test_vast_${diar_name}}/xvector.scp > $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp
    
    utils/combine_data.sh data/sitw_sre18_dev_vast_${diar_name} data/sitw_dev_${diar_name} data/sre18_dev_vast_${diar_name} 
    mkdir -p $xvector_dir/sitw_sre18_dev_vast_${diar_name}
    cat $xvector_dir/sitw_dev_${diar_name}/xvector.scp $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp > $xvector_dir/sitw_sre18_dev_vast_${diar_name}/xvector.scp

fi

if [ $stage -le 21 ]; then
    # steps_be/train_vid_be_v1.sh --cmd "$train_cmd" \
    # 				--lda_dim $vid_lda_dim \
    # 				--plda_type $plda_vid_type \
    # 				--y_dim $plda_vid_y_dim --z_dim $plda_vid_z_dim \
    # 				$xvector_dir/$plda_vid_data/xvector.scp \
    # 				data/$plda_vid_data \
    # 				$xvector_dir/sitw_dev/xvector.scp \
    # 				data/sitw_dev \
    # 				$be_vid_dir &


    # steps_be/train_tel_be_v1.sh --cmd "$train_cmd" \
    # 				--lda_dim $tel_lda_dim \
    # 				--plda_type $plda_tel_type \
    # 				--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    # 				--w_mu $w_mu --w_B $w_B --w_W $w_W \
    # 				$xvector_dir/$plda_tel_data/xvector.scp \
    # 				data/$plda_tel_data \
    # 				$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    # 				data/sre18_dev_unlabeled \
    # 				$sre18_dev_meta $be_tel_dir &

    steps_be/train_vid_be_v3.sh --cmd "$train_cmd" \
				--lda_dim $vid_lda_dim \
				--plda_type $plda_vid_type \
				--y_dim $plda_vid_y_dim --z_dim $plda_vid_z_dim \
				$xvector_dir/$plda_vid_data/xvector.scp \
				data/$plda_vid_data \
				$xvector_dir/sitw_dev_${diar_name}/xvector.scp \
				data/sitw_dev_${diar_name} \
				$xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
				data/sre18_dev_vast_${diar_name} \
				$be_vid_dir &

    steps_be/train_tel_be_v2.1.sh --cmd "$train_cmd" \
    				--lda_dim $tel_lda_dim \
    				--plda_type $plda_tel_type \
    				--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    				--w_mu1 $w_mu1 --w_B1 $w_B1 --w_W1 $w_W1 \
    				--w_mu2 $w_mu2 --w_B2 $w_B2 --w_W2 $w_W2 --num_spks $num_spks \
    				$xvector_dir/$plda_tel_data/xvector.scp \
    				data/$plda_tel_data \
    				$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    				$sre18_dev_meta $be_tel_dir &
    
    # steps_be/train_tel_be_v5.sh --cmd "$train_cmd" \
    # 				--lda_dim $tel_lda_dim \
    # 				--plda_type $plda_tel_type \
    # 				--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    # 				--w_mu1 $w_mu1 --w_B1 $w_B1 --w_W1 $w_W1 \
    # 				--w_mu2 $w_mu2 --w_B2 $w_B2 --w_W2 $w_W2 --num_spks $num_spks \
    # 				--w_mu3 $w_mu3 --w_B3 $w_B3 --w_W3 $w_W3 \
    # 				--w_mu4 $w_mu4 --w_B4 $w_B4 --w_W4 $w_W4 \
    # 				$xvector_dir/$plda_tel_data/xvector.scp \
    # 				data/$plda_tel_data \
    # 				$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    # 				data/sre18_dev_unlabeled \
    # 				$sre18_dev_meta $be_tel_dir &

    # steps_be/train_tel_be_v4.sh --cmd "$train_cmd" \
    # 				--lda_dim $tel_lda_dim \
    # 				--plda_type $plda_tel_type \
    # 				--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    # 				--w_mu1 $w_mu1 --w_B1 $w_B1 --w_W1 $w_W1 \
    # 				--w_mu2 $w_mu2 --w_B2 $w_B2 --w_W2 $w_W2 --num_spks $num_spks \
    # 				--w_mu3 $w_mu3 --w_B3 $w_B3 --w_W3 $w_W3 --lda_dim_adapt $cmn2_lda_dim \
    # 				$xvector_dir/$plda_tel_data/xvector.scp \
    # 				data/$plda_tel_data \
    # 				$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    # 				data/sre18_dev_unlabeled \
    # 				$sre18_dev_meta $be_tel_dir &

    wait

fi


if [ $stage -le 22 ];then

    #SITW
    echo "SITW no-diarization"

    steps_be/eval_vid_be_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
			       $sitw_eval_trials_core \
			       data/sitw_eval_enroll/utt2spk \
			       $xvector_dir/sitw_eval/xvector.scp \
			       $be_vid_dir/lda_lnorm_adapt.h5 \
			       $be_vid_dir/plda.h5 \
			       $score_plda_out_dir/sitw_eval_core_scores &


    steps_be/eval_vid_be_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
			       $sitw_eval_trials_core_multi \
			       data/sitw_eval_enroll/utt2spk \
			       $xvector_dir/sitw_eval/xvector.scp \
			       $be_vid_dir/lda_lnorm_adapt.h5 \
			       $be_vid_dir/plda.h5 \
			       $score_plda_out_dir/sitw_eval_core_multi_scores &

    steps_be/eval_vid_be_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
			       $sitw_eval_trials_assist_core \
			       data/sitw_eval_enroll/utt2spk \
			       $xvector_dir/sitw_eval/xvector.scp \
			       $be_vid_dir/lda_lnorm_adapt.h5 \
			       $be_vid_dir/plda.h5 \
			       $score_plda_out_dir/sitw_eval_assist_core_scores &

    steps_be/eval_vid_be_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
			       $sitw_eval_trials_assist_multi \
			       data/sitw_eval_enroll/utt2spk \
			       $xvector_dir/sitw_eval/xvector.scp \
			       $be_vid_dir/lda_lnorm_adapt.h5 \
			       $be_vid_dir/plda.h5 \
			       $score_plda_out_dir/sitw_eval_assist_multi_scores &
    wait

    local/score_sitw_eval.sh $score_plda_out_dir
    
fi

if [ $stage -le 23 ]; then

    mkdir -p $xvector_dir/sre18_dev_cmn2
    cat $xvector_dir/sre18_dev_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_dev_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre18_dev_vast
    cat $xvector_dir/sre18_dev_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_dev_vast/xvector.scp

    mkdir -p $xvector_dir/sre18_eval_cmn2
    cat $xvector_dir/sre18_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_eval_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre18_eval_vast
    cat $xvector_dir/sre18_eval_{enroll,test}_vast/xvector.scp > $xvector_dir/sre18_eval_vast/xvector.scp

    
    #SRE18
    echo "SRE18 no-diarization"

    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type \
			       $sre18_dev_trials_cmn2 \
			       data/sre18_dev_enroll_cmn2/utt2spk \
			       $xvector_dir/sre18_dev_cmn2/xvector.scp \
			       $be_tel_dir/lda_lnorm_adapt.h5 \
			       $be_tel_dir/plda_adapt2.h5 \
			       $score_plda_out_dir/sre18_dev_cmn2_scores &

    
    steps_be/eval_vid_be_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
    			       $sre18_dev_trials_vast \
    			       data/sre18_dev_enroll_vast/utt2spk \
    			       $xvector_dir/sre18_dev_vast/xvector.scp \
    			       $be_vid_dir/lda_lnorm_adapt2.h5 \
    			       $be_vid_dir/plda.h5 \
    			       $score_plda_out_dir/sre18_dev_vast_scores &


    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type \
			       $sre18_eval_trials_cmn2 \
			       data/sre18_eval_enroll_cmn2/utt2spk \
			       $xvector_dir/sre18_eval_cmn2/xvector.scp \
			       $be_tel_dir/lda_lnorm_adapt.h5 \
			       $be_tel_dir/plda_adapt2.h5 \
			       $score_plda_out_dir/sre18_eval_cmn2_scores &

    
    steps_be/eval_vid_be_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
    			       $sre18_eval_trials_vast \
    			       data/sre18_eval_enroll_vast/utt2spk \
    			       $xvector_dir/sre18_eval_vast/xvector.scp \
    			       $be_vid_dir/lda_lnorm_adapt2.h5 \
    			       $be_vid_dir/plda.h5 \
    			       $score_plda_out_dir/sre18_eval_vast_scores &

    wait

    #local/score_sre18_old.sh $score_plda_out_dir $score_plda_out_dir
    local/score_sre18.sh $sre18_dev_root dev $score_plda_out_dir/sre18_dev_cmn2_scores $score_plda_out_dir/sre18_dev_vast_scores $score_dir/sre18_plda_out
    local/score_sre18.sh $sre18_eval_root eval $score_plda_out_dir/sre18_eval_cmn2_scores $score_plda_out_dir/sre18_eval_vast_scores $score_dir/sre18_plda_out

fi

if [ $stage -le 24 ];then
    #local/calibration_sre18_v1.sh $score_plda_out_dir $score_plda_out_dir
    #local/score_sitw_eval.sh ${score_plda_out_dir}_cal_v1
    #local/score_sre18_old.sh ${score_plda_out_dir}_cal_v1 ${score_plda_out_dir}_cal_v1
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_out_dir}_cal_v1/sre18_dev_cmn2_scores ${score_plda_out_dir}_cal_v1/sre18_dev_vast_scores $score_dir/sre18_plda_out_cal_v1
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_out_dir}_cal_v1/sre18_eval_cmn2_scores ${score_plda_out_dir}_cal_v1/sre18_eval_vast_scores $score_dir/sre18_plda_out_cal_v1
    exit
fi


if [ $stage -le 25 ];then

    mkdir -p $xvector_dir/sitw_eval_${diar_name}
    cat $xvector_dir/sitw_eval_{enroll,test_${diar_name}}/xvector.scp > $xvector_dir/sitw_eval_${diar_name}/xvector.scp
    
    #SITW
    echo "SITW with diarization"

    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
			       $sitw_eval_trials_core $sitw_eval_trials_core_diar \
			       data/sitw_eval_enroll/utt2spk \
			       $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
			       data/sitw_eval_test_${diar_name}/utt2orig \
			       $be_vid_dir/lda_lnorm_adapt.h5 \
			       $be_vid_dir/plda.h5 \
			       ${score_plda_out_dir}_${diar_name}/sitw_eval_core_scores


    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
			       $sitw_eval_trials_core_multi $sitw_eval_trials_core_multi_diar \
			       data/sitw_eval_enroll/utt2spk \
			       $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
			       data/sitw_eval_test_${diar_name}/utt2orig \
			       $be_vid_dir/lda_lnorm_adapt.h5 \
			       $be_vid_dir/plda.h5 \
			       ${score_plda_out_dir}_${diar_name}/sitw_eval_core_multi_scores &

    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
			       $sitw_eval_trials_assist_core $sitw_eval_trials_assist_core_diar \
			       data/sitw_eval_enroll/utt2spk \
			       $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
			       data/sitw_eval_test_${diar_name}/utt2orig \
			       $be_vid_dir/lda_lnorm_adapt.h5 \
			       $be_vid_dir/plda.h5 \
			       ${score_plda_out_dir}_${diar_name}/sitw_eval_assist_core_scores &

    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
				    $sitw_eval_trials_assist_multi $sitw_eval_trials_assist_multi_diar \
				    data/sitw_eval_enroll/utt2spk \
				    $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
				    data/sitw_eval_test_${diar_name}/utt2orig \
				    $be_vid_dir/lda_lnorm_adapt.h5 \
				    $be_vid_dir/plda.h5 \
				    ${score_plda_out_dir}_${diar_name}/sitw_eval_assist_multi_scores &
    wait

    local/score_sitw_eval.sh ${score_plda_out_dir}_${diar_name}
    
fi


if [ $stage -le 26 ]; then

    mkdir -p $xvector_dir/sre18_dev_vast_${diar_name}
    cat $xvector_dir/sre18_dev_{enroll_vast,test_vast_${diar_name}}/xvector.scp > $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp

    mkdir -p $xvector_dir/sre18_eval_vast_${diar_name}
    cat $xvector_dir/sre18_eval_{enroll_vast,test_vast_${diar_name}}/xvector.scp > $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp

    
    #SRE18
    # Get results using the out-of-domain PLDA model with diarization
    echo "SRE18 with diarization"
    
    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
    				    $sre18_dev_trials_vast $sre18_dev_trials_vast_diar \
    				    data/sre18_dev_enroll_vast/utt2spk \
    				    $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
				    data/sre18_dev_test_vast_${diar_name}/utt2orig \
    				    $be_vid_dir/lda_lnorm_adapt2.h5 \
    				    $be_vid_dir/plda.h5 \
    				    ${score_plda_out_dir}_${diar_name}/sre18_dev_vast_scores

    steps_be/eval_vid_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type \
    				    $sre18_eval_trials_vast $sre18_eval_trials_vast_diar \
    				    data/sre18_eval_enroll_vast/utt2spk \
    				    $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp \
				    data/sre18_eval_test_vast_${diar_name}/utt2orig \
    				    $be_vid_dir/lda_lnorm_adapt2.h5 \
    				    $be_vid_dir/plda.h5 \
    				    ${score_plda_out_dir}_${diar_name}/sre18_eval_vast_scores

    
    #local/score_sre18_old.sh $score_plda_out_dir ${score_plda_out_dir}_${diar_name}
    local/score_sre18.sh $sre18_dev_root dev $score_plda_out_dir/sre18_dev_cmn2_scores ${score_plda_out_dir}_${diar_name}/sre18_dev_vast_scores $score_dir/sre18_plda_out_${diar_name}
    local/score_sre18.sh $sre18_eval_root eval $score_plda_out_dir/sre18_eval_cmn2_scores ${score_plda_out_dir}_${diar_name}/sre18_eval_vast_scores $score_dir/sre18_plda_out_${diar_name}

fi

if [ $stage -le 27 ];then
    #local/calibration_sre18_v1.sh $score_plda_out_dir ${score_plda_out_dir}_${diar_name}
    #local/score_sitw_eval.sh ${score_plda_out_dir}_${diar_name}_cal_v1
    #local/score_sre18_old.sh ${score_plda_out_dir}_cal_v1 ${score_plda_out_dir}_${diar_name}_cal_v1
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_out_dir}_cal_v1/sre18_dev_cmn2_scores ${score_plda_out_dir}_${diar_name}_cal_v1/sre18_dev_vast_scores $score_dir/sre18_plda_out_${diar_name}_cal_v1
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_out_dir}_cal_v1/sre18_eval_cmn2_scores ${score_plda_out_dir}_${diar_name}_cal_v1/sre18_eval_vast_scores $score_dir/sre18_plda_out_${diar_name}_cal_v1
    exit
fi


if [ $stage -le 32 ];then

    #SITW
    echo "SITW S-Norm"

    steps_be/eval_vid_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
				     $sitw_eval_trials_core \
				     data/sitw_eval_enroll/utt2spk \
				     $xvector_dir/sitw_eval/xvector.scp \
				     data/${coh_vid_data}/utt2spk \
				     $xvector_dir/${coh_vid_data}/xvector.scp \
				     $be_vid_dir/lda_lnorm_adapt.h5 \
				     $be_vid_dir/plda.h5 \
				     ${score_plda_out_dir}_snorm/sitw_eval_core_scores &


    steps_be/eval_vid_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
				     $sitw_eval_trials_core_multi \
				     data/sitw_eval_enroll/utt2spk \
				     $xvector_dir/sitw_eval/xvector.scp \
				     data/${coh_vid_data}/utt2spk \
				     $xvector_dir/${coh_vid_data}/xvector.scp \
				     $be_vid_dir/lda_lnorm_adapt.h5 \
				     $be_vid_dir/plda.h5 \
				     ${score_plda_out_dir}_snorm/sitw_eval_core_multi_scores &

    steps_be/eval_vid_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
				     $sitw_eval_trials_assist_core \
				     data/sitw_eval_enroll/utt2spk \
				     $xvector_dir/sitw_eval/xvector.scp \
				     data/${coh_vid_data}/utt2spk \
				     $xvector_dir/${coh_vid_data}/xvector.scp \
				     $be_vid_dir/lda_lnorm_adapt.h5 \
				     $be_vid_dir/plda.h5 \
				     ${score_plda_out_dir}_snorm/sitw_eval_assist_core_scores &

    steps_be/eval_vid_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
				     $sitw_eval_trials_assist_multi \
				     data/sitw_eval_enroll/utt2spk \
				     $xvector_dir/sitw_eval/xvector.scp \
				     data/${coh_vid_data}/utt2spk \
				     $xvector_dir/${coh_vid_data}/xvector.scp \
				     $be_vid_dir/lda_lnorm_adapt.h5 \
				     $be_vid_dir/plda.h5 \
				     ${score_plda_out_dir}_snorm/sitw_eval_assist_multi_scores &
    wait

    local/score_sitw_eval.sh ${score_plda_out_dir}_snorm
    
fi



if [ $stage -le 33 ]; then

    #SRE18
    echo "SRE18 S-Norm"

    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type --ncoh $tel_ncoh \
				     $sre18_dev_trials_cmn2 \
				     data/sre18_dev_enroll_cmn2/utt2spk \
				     $xvector_dir/sre18_dev_cmn2/xvector.scp \
				     data/${coh_tel_data}/utt2spk \
				     $xvector_dir/${coh_tel_data}/xvector.scp \
				     $be_tel_dir/lda_lnorm_adapt.h5 \
				     $be_tel_dir/plda_adapt2.h5 \
				     ${score_plda_out_dir}_snorm/sre18_dev_cmn2_scores &

    
    steps_be/eval_vid_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vast_ncoh --ncoh_discard 7 \
    				     $sre18_dev_trials_vast \
    				     data/sre18_dev_enroll_vast/utt2spk \
    				     $xvector_dir/sre18_dev_vast/xvector.scp \
				     data/${coh_vast_data}/utt2spk \
				     $xvector_dir/${coh_vast_data}/xvector.scp \
    				     $be_vid_dir/lda_lnorm_adapt2.h5 \
    				     $be_vid_dir/plda.h5 \
    				     ${score_plda_out_dir}_snorm/sre18_dev_vast_scores &

    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type --ncoh $tel_ncoh \
				     $sre18_eval_trials_cmn2 \
				     data/sre18_eval_enroll_cmn2/utt2spk \
				     $xvector_dir/sre18_eval_cmn2/xvector.scp \
				     data/${coh_tel_data}/utt2spk \
				     $xvector_dir/${coh_tel_data}/xvector.scp \
				     $be_tel_dir/lda_lnorm_adapt.h5 \
				     $be_tel_dir/plda_adapt2.h5 \
				     ${score_plda_out_dir}_snorm/sre18_eval_cmn2_scores &

    
    steps_be/eval_vid_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vast_ncoh \
    				     $sre18_eval_trials_vast \
    				     data/sre18_eval_enroll_vast/utt2spk \
    				     $xvector_dir/sre18_eval_vast/xvector.scp \
				     data/${coh_vast_data}/utt2spk \
				     $xvector_dir/${coh_vast_data}/xvector.scp \
    				     $be_vid_dir/lda_lnorm_adapt2.h5 \
    				     $be_vid_dir/plda.h5 \
    				     ${score_plda_out_dir}_snorm/sre18_eval_vast_scores &

    
    wait

    #local/score_sre18_old.sh ${score_plda_out_dir}_snorm ${score_plda_out_dir}_snorm
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_out_dir}_snorm/sre18_dev_cmn2_scores ${score_plda_out_dir}_snorm/sre18_dev_vast_scores $score_dir/sre18_plda_out_snorm
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_out_dir}_snorm/sre18_eval_cmn2_scores ${score_plda_out_dir}_snorm/sre18_eval_vast_scores $score_dir/sre18_plda_out_snorm

fi

if [ $stage -le 34 ];then
    #local/calibration_sre18_v1.sh ${score_plda_out_dir}_snorm ${score_plda_out_dir}_snorm
    #local/score_sitw_eval.sh ${score_plda_out_dir}_snorm_cal_v1
    #local/score_sre18_old.sh ${score_plda_out_dir}_snorm_cal_v1 ${score_plda_out_dir}_snorm_cal_v1
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_out_dir}_snorm_cal_v1/sre18_dev_cmn2_scores ${score_plda_out_dir}_snorm_cal_v1/sre18_dev_vast_scores $score_dir/sre18_plda_out_snorm_cal_v1
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_out_dir}_snorm_cal_v1/sre18_eval_cmn2_scores ${score_plda_out_dir}_snorm_cal_v1/sre18_eval_vast_scores $score_dir/sre18_plda_out_snorm_cal_v1
    exit
fi




if [ $stage -le 35 ];then

    #SITW
    # Get results using diarization and s-norm
    echo "SITW S-Norm with diarization"

    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
					  $sitw_eval_trials_core $sitw_eval_trials_core_diar \
					  data/sitw_eval_enroll/utt2spk \
					  $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
					  data/sitw_eval_test_${diar_name}/utt2orig \
					  data/${coh_vid_data}/utt2spk \
					  $xvector_dir/${coh_vid_data}/xvector.scp \
					  $be_vid_dir/lda_lnorm_adapt.h5 \
					  $be_vid_dir/plda.h5 \
					  ${score_plda_out_dir}_${diar_name}_snorm/sitw_eval_core_scores &
    

    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
				     $sitw_eval_trials_core_multi $sitw_eval_trials_core_multi_diar \
				     data/sitw_eval_enroll/utt2spk \
				     $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
				     data/sitw_eval_test_${diar_name}/utt2orig \
				     data/${coh_vid_data}/utt2spk \
				     $xvector_dir/${coh_vid_data}/xvector.scp \
				     $be_vid_dir/lda_lnorm_adapt.h5 \
				     $be_vid_dir/plda.h5 \
				     ${score_plda_out_dir}_${diar_name}_snorm/sitw_eval_core_multi_scores &

    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
					  $sitw_eval_trials_assist_core $sitw_eval_trials_assist_core_diar \
					  data/sitw_eval_enroll/utt2spk \
					  $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
					  data/sitw_eval_test_${diar_name}/utt2orig \
					  data/${coh_vid_data}/utt2spk \
					  $xvector_dir/${coh_vid_data}/xvector.scp \
					  $be_vid_dir/lda_lnorm_adapt.h5 \
					  $be_vid_dir/plda.h5 \
					  ${score_plda_out_dir}_${diar_name}_snorm/sitw_eval_assist_core_scores &
    
    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vid_ncoh \
					  $sitw_eval_trials_assist_multi $sitw_eval_trials_assist_multi_diar \
					  data/sitw_eval_enroll/utt2spk \
					  $xvector_dir/sitw_eval_${diar_name}/xvector.scp \
					  data/sitw_eval_test_${diar_name}/utt2orig \
					  data/${coh_vid_data}/utt2spk \
					  $xvector_dir/${coh_vid_data}/xvector.scp \
					  $be_vid_dir/lda_lnorm_adapt.h5 \
					  $be_vid_dir/plda.h5 \
					  ${score_plda_out_dir}_${diar_name}_snorm/sitw_eval_assist_multi_scores &
    wait

    local/score_sitw_eval.sh ${score_plda_out_dir}_${diar_name}_snorm
    
fi



if [ $stage -le 36 ]; then

    #SRE18
    # Get results using the out-of-domain PLDA model.
    echo "SRE18 S-Norm with diarization"

    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vast_ncoh --ncoh_discard 7 \
    					  $sre18_dev_trials_vast $sre18_dev_trials_vast_diar \
    					  data/sre18_dev_enroll_vast/utt2spk \
    					  $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
					  data/sre18_dev_test_vast_${diar_name}/utt2orig \
					  data/${coh_vast_data}/utt2spk \
					  $xvector_dir/${coh_vast_data}/xvector.scp \
    					  $be_vid_dir/lda_lnorm_adapt2.h5 \
    					  $be_vid_dir/plda.h5 \
    					  ${score_plda_out_dir}_${diar_name}_snorm/sre18_dev_vast_scores &

    steps_be/eval_vid_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_vid_type --ncoh $vast_ncoh \
    					  $sre18_eval_trials_vast $sre18_eval_trials_vast_diar \
    					  data/sre18_eval_enroll_vast/utt2spk \
    					  $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp \
					  data/sre18_eval_test_vast_${diar_name}/utt2orig \
					  data/${coh_vast_data}/utt2spk \
					  $xvector_dir/${coh_vast_data}/xvector.scp \
    					  $be_vid_dir/lda_lnorm_adapt2.h5 \
    					  $be_vid_dir/plda.h5 \
    					  ${score_plda_out_dir}_${diar_name}_snorm/sre18_eval_vast_scores &

    wait

    #local/score_sre18_old.sh ${score_plda_out_dir}_snorm ${score_plda_out_dir}_${diar_name}_snorm
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_out_dir}_snorm/sre18_dev_cmn2_scores ${score_plda_out_dir}_${diar_name}_snorm/sre18_dev_vast_scores $score_dir/sre18_plda_out_${diar_name}_snorm
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_out_dir}_snorm/sre18_eval_cmn2_scores ${score_plda_out_dir}_${diar_name}_snorm/sre18_eval_vast_scores $score_dir/sre18_plda_out_${diar_name}_snorm

fi

if [ $stage -le 37 ];then
    #local/calibration_sre18_v1.sh ${score_plda_out_dir}_snorm ${score_plda_out_dir}_${diar_name}_snorm
    #local/score_sitw_eval.sh ${score_plda_out_dir}_${diar_name}_snorm_cal_v1
    #local/score_sre18_old.sh ${score_plda_out_dir}_snorm_cal_v1 ${score_plda_out_dir}_${diar_name}_snorm_cal_v1
    local/score_sre18.sh $sre18_dev_root dev ${score_plda_out_dir}_snorm_cal_v1/sre18_dev_cmn2_scores ${score_plda_out_dir}_${diar_name}_snorm_cal_v1/sre18_dev_vast_scores $score_dir/sre18_plda_out_${diar_name}_snorm_cal_v1
    local/score_sre18.sh $sre18_eval_root eval ${score_plda_out_dir}_snorm_cal_v1/sre18_eval_cmn2_scores ${score_plda_out_dir}_${diar_name}_snorm_cal_v1/sre18_eval_vast_scores $score_dir/sre18_plda_out_${diar_name}_snorm_cal_v1
    exit
fi


    
exit
