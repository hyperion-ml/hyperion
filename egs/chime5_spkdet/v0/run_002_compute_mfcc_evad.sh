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
nodes=c0
storage_name=$(date +'%m_%d_%H_%M')
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
vaddir_gt=`pwd`/vad_gt

stage=1

. parse_options.sh || exit 1;

# Make filterbanks and compute the energy-based VAD for each dataset

if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
	dir_name=$USER/hyp-data/voices_challenge/v1/$storage_name/mfcc/storage
	if [ "$nodes" == "b0" ];then
	    utils/create_split_dir.pl \
			    utils/create_split_dir.pl \
		/export/b{04,05,06,07}/$dir_name $mfccdir/storage
	elif [ "$nodes" == "b1" ];then
	    utils/create_split_dir.pl \
		/export/b{14,15,16,17}/$dir_name $mfccdir/storage
	else
	    utils/create_split_dir.pl \
		/export/c{06,07,08,09}/$dir_name $mfccdir/storage
	fi
    fi
fi

#Train datasets
if [ $stage -le 2 ];then 
    #for name in voxceleb1cat voxceleb2cat_train sitw_train chime5_spkdet_enroll chime5_spkdet_test
    for name in chime5_spkdet_test
    do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj 40 --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	steps_fe/compute_vad_decision.sh --nj 30 --cmd "$train_cmd" \
					 data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done

    for name in chime5_spkdet_enroll
    do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj 1 --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	steps_fe/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
					 data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done

    
fi

#Combine voxceleb
if [ $stage -le 3 ];then 
  utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb data/voxceleb1cat data/voxceleb2cat_train
  utils/fix_data_dir.sh data/voxceleb

fi

if [ $stage -le 4 ];then
    for name in chime5_spkdet_enroll
    do
	hyp_utils/segments_to_bin_vad.sh --nj 5 data/$name/diarization_segments data/$name $vaddir_gt
	utils/fix_data_dir.sh data/$name
    done
    for name in chime5_spkdet_test
    do
	rm -rf data/${name}_gtvad
	cp -r data/$name data/${name}_gtvad
	name=${name}_gtvad
	hyp_utils/segments_to_bin_vad.sh --nj 5 data/$name/diarization_segments data/$name $vaddir_gt
	utils/fix_data_dir.sh data/$name
    done
fi
