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
nodes=fs01
storage_name=$(date +'%m_%d_%H_%M')
mfccdir=`pwd`/exp/mfcc
vaddir=`pwd`/exp/mfcc
vaddir_gt=`pwd`/exp/vad_gt

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;

  # Make filterbanks and compute the energy-based VAD for each dataset

if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
	dir_name=$USER/hyp-data/sre19-av/v1/$storage_name/mfcc/storage
	if [ "$nodes" == "b0" ];then
	    utils/create_split_dir.pl \
			    utils/create_split_dir.pl \
		/export/b{04,05,06,07}/$dir_name $mfccdir/storage
	elif [ "$nodes" == "b1" ];then
	    utils/create_split_dir.pl \
		/export/b{14,15,16,17}/$dir_name $mfccdir/storage
	elif [ "$nodes" == "c0" ];then
	    utils/create_split_dir.pl \
		/export/c{06,07,08,09}/$dir_name $mfccdir/storage
	elif [ "$nodes" == "fs01" ];then
	    utils/create_split_dir.pl \
		/export/fs01/$dir_name $mfccdir/storage
	else
	    echo "we don't distribute data between multiple machines"
	fi
    fi
fi

#Train datasets
if [ $stage -le 2 ];then 
    for name in voxceleb1cat voxceleb2cat_train \
    	sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test \
        sre18_dev_test_vast sre18_eval_test_vast \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_enroll janus_dev_test_core janus_eval_enroll janus_eval_test_core 
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 40 ? $num_spk:40))
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj $nj --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc/$name $mfccdir
	utils/fix_data_dir.sh data/${name}
	steps_fe/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
					 data/${name} exp/make_vad/$name $vaddir
	utils/fix_data_dir.sh data/${name}
    done

    for name in sre18_dev_enroll_vast sre18_eval_enroll_vast sre19_av_a_dev_enroll sre19_av_a_eval_enroll
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 40 ? $num_spk:40))
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj $nj --cmd "$train_cmd" \
    			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	local/sre18_diar_to_vad.sh data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done

fi


if [ $stage -le 3 ];then 
  utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb data/voxceleb1cat data/voxceleb2cat_train
  utils/fix_data_dir.sh data/voxceleb
fi

if [ $stage -le 4 ];then
    for name in dihard2_train_dev dihard2_train_eval
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 40 ? $num_spk:40))
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj $nj --cmd "$train_cmd" \
    			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	hyp_utils/rttm_to_bin_vad.sh --nj 5 data/$name/vad.rttm data/$name $vaddir_gt
	utils/fix_data_dir.sh data/${name}
    done

fi

if [ $stage -le 5 ];then 
  utils/combine_data.sh --extra-files "utt2num_frames" data/dihard2_train data/dihard2_train_dev data/dihard2_train_eval
  utils/fix_data_dir.sh data/dihard2_train
fi

