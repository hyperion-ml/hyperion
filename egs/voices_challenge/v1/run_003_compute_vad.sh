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
nnet_vaddir=`pwd`/vad

stage=1

. parse_options.sh || exit 1;

# Compute VAD with Aspire model

if [ $stage -le 1 ];then
    # download model from Dan's page
    if [ ! -d vad_model ];then
	wget -P ./vad_model http://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
	tar -C ./vad_model -xzvf ./vad_model/0004_tdnn_stats_asr_sad_1a.tar.gz
    fi
fi

vad_model_dir=vad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a
vad_mfcc_conf=vad_model/conf/mfcc_hires.conf

if [ $stage -le 2 ];then 
    for name in voxceleb1cat voxceleb2cat_train sitw_train
    do
	steps_fe/detect_speech_activity.sh --nj 40 --cmd "$train_cmd"  \
					   --extra-left-context 79 --extra-right-context 21 \
					   --extra-left-context-initial 0 --extra-right-context-final 0 \
					   --frames-per-chunk 150 --mfcc-config $vad_mfcc_conf \
					   --dur-read-entire-file true \
					   data/$name $vad_model_dir \
					   mfccdir_vad exp/make_tdnn_vad data/${name}
	if [ ! -f data/$name/utt2num_frames ];then
	    cp data/${name}_hires/utt2num_frames data/$name
	fi
	steps_fe/segments2vad.sh data/$name data/${name}_seg $nnet_vaddir
	hyp_utils/remove_utts_wo_vad.sh data/$name
	utils/fix_data_dir.sh data/${name}
    done
  
fi


if [ $stage -le 3 ];then 
  utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb data/voxceleb1cat data/voxceleb2cat_train
  utils/fix_data_dir.sh data/voxceleb
  exit
fi

if [ $stage -le 4 ];then 

    for name in voices19_challenge_dev_enroll voices19_challenge_dev_test voices19_challenge_eval_enroll voices19_challenge_eval_test
    do
	steps_fe/detect_speech_activity.sh --nj 40 --cmd "$train_cmd"  \
					   --extra-left-context 79 --extra-right-context 21 \
					   --extra-left-context-initial 0 --extra-right-context-final 0 \
					   --frames-per-chunk 150 --mfcc-config $vad_mfcc_conf \
					   --dur-read-entire-file true \
					   data/$name $vad_model_dir \
					   mfccdir_vad exp/make_tdnn_vad data/${name}
	if [ ! -f data/$name/utt2num_frames ];then
	    cp data/${name}_hires/utt2num_frames data/$name
	fi
	steps_fe/segments2vad.sh data/$name data/${name}_seg $nnet_vaddir
	utils/fix_data_dir.sh data/${name}
    done
fi

if [ $stage -le 5 ];then
    
    #fix vimals vad in eval
    for name in voices19_challenge_eval_test
    do
	if [ ! -f data/$name/vad.nn.scp ];then
	    mv data/$name/vad.scp data/$name/vad.nn.scp
	fi
	# compute energy VAD
    	steps_fe/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
					 data/${name} exp/make_vad vad_e
	mv data/$name/vad.scp data/$name/vad.e.scp
	#Put energy VAD for utts where NN vad is missing.
	awk -v fvv=data/$name/vad.nn.scp 'BEGIN{
            while(getline < fvv)
            {
                 vv[$1]=$0
            }
        } 
        { if($1 in vv){ print vv[$1]} else { print $0 }}' data/$name/vad.e.scp > data/$name/vad.scp
	utils/fix_data_dir.sh data/${name}
    done

fi

