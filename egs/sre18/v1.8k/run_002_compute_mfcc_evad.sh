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

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;

# Make MFCC and compute the energy-based VAD for each dataset

if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
	dir_name=$USER/hyp-data/sre18/v1.8k/$storage_name/mfcc/storage
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
    
    for name in sre_tel sre_phnmic swbd voxceleb1cat voxceleb2cat_train \
    			sre16_eval_enroll sre16_eval_test sre16_major sre16_minor \
    			sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test sitw_train_dev\
    			sre18_dev_unlabeled sre18_dev_enroll_cmn2 sre18_dev_test_cmn2 \
    			sre18_eval_enroll_cmn2 sre18_eval_test_cmn2 sre18_eval_test_vast;
    do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_8k.conf --nj 40 --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	steps_fe/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
					 data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done

    for name in sre16_dev_enroll sre16_dev_test sre18_dev_test_vast; do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_8k.conf --nj 5 --cmd "$train_cmd" \
    			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	steps_fe/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
    					 data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done
  
  
    for name in sre18_dev_enroll_vast; do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_8k.conf --nj 1 --cmd "$train_cmd" \
    			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	local/sre18_diar_to_vad.sh data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done

    for name in sre18_eval_enroll_vast; do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_8k.conf --nj 8 --cmd "$train_cmd" \
    			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	local/sre18_diar_to_vad.sh data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done

fi


if [ $stage -le 3 ];then 
  utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre_tel data/swbd data/sre_tel
  utils/fix_data_dir.sh data/swbd_sre_tel
  utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb data/voxceleb1cat data/voxceleb2cat_train
  utils/fix_data_dir.sh data/voxceleb
fi

