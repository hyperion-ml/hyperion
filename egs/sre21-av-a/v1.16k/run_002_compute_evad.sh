#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e
nodes=fs01
storage_name=$(date +'%m_%d_%H_%M')
vaddir=`pwd`/exp/vad_e
vad_config=conf/vad_16k.yaml

stage=1
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file


if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $vaddir/storage ]; then
	dir_name=$USER/hyp-data/sre21/v1.16k/$storage_name/vad/storage
	if [ "$nodes" == "b0" ];then
	    utils/create_split_dir.pl \
			    utils/create_split_dir.pl \
		/export/b{04,05,06,07}/$dir_name $vaddir/storage
	elif [ "$nodes" == "b1" ];then
	    utils/create_split_dir.pl \
		/export/b{14,15,16,17}/$dir_name $vaddir/storage
	elif [ "$nodes" == "c0" ];then
	    utils/create_split_dir.pl \
		/export/c{06,07,08,09}/$dir_name $vaddir/storage
	elif [ "$nodes" == "fs01" ];then
	    utils/create_split_dir.pl \
		/export/fs01/$dir_name $vaddir/storage
	else
	    echo "we don't distribute data between multiple machines"
	fi
    fi
fi

# VAD Train/Test Datasets
if [ $stage -le 2 ];then 
  for name in voxcelebcat \
  		sre_cts_superset_16k_trn \
  		sre_cts_superset_16k_dev \
  		sre16_eval40_yue_enroll \
  		sre16_eval40_yue_test \
  		sre16_eval_tr60_tgl \
  		sre16_eval_tr60_yue \
  		sre16_train_dev_ceb \
  		sre16_train_dev_cmn \
		sre21_audio_dev_enroll \
		sre21_audio_dev_test \
		sre21_audio-visual_dev_test \
		sre21_audio_eval_enroll \
		sre21_audio_eval_test \
		sre21_audio-visual_eval_test
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 40 ? $num_spk:40))
    hyp_utils/feats/make_evad.sh --write-utt2num-frames true \
				 --vad-config $vad_config --nj $nj --cmd "$train_cmd" \
				 data/${name} exp/make_vad/$name $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

# Copy VAD from 16k to 8k datasets
if [ $stage -le 3 ];then
  for name in voxcelebcat
  do
    awk '{ print $1"-8k",$2}' data/$name/vad.scp > data/${name}_8k/vad.scp
  done
fi

# #Enroll multi-speaker Datasets with time marks
# if [ $stage -le 3 ];then 
#     for name in sre18_dev_enroll_vast sre18_eval_enroll_vast sre19_av_a_dev_enroll sre19_av_a_eval_enroll
#     do
# 	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
# 	nj=$(($num_spk < 40 ? $num_spk:40))
# 	# we just run energy vad to get the utt2num_frames file
# 	hyp_utils/feats/make_evad.sh --write-utt2num-frames true \
# 	    --vad-config $vad_config --nj $nj --cmd "$train_cmd" \
# 	    data/${name} exp/make_vad/$name $vaddir
# 	utils/fix_data_dir.sh data/${name}
# 	local/sre18_diar_to_vad.sh data/${name} exp/make_vad $vaddir
# 	utils/fix_data_dir.sh data/${name}
#     done
# fi

# #Dihard Datasets
# if [ $stage -le 4 ];then
#     for name in dihard2_train_dev dihard2_train_eval
#     do
# 	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
# 	nj=$(($num_spk < 40 ? $num_spk:40))
# 	# we just run energy vad to get the utt2num_frames file
# 	hyp_utils/feats/make_evad.sh --write-utt2num-frames true \
# 	    --vad-config $vad_config --nj $nj --cmd "$train_cmd" \
# 	    data/${name} exp/make_vad/$name $vaddir
# 	hyp_utils/rttm_to_bin_vad.sh --nj 5 data/$name/vad.rttm data/$name $vaddir
# 	utils/fix_data_dir.sh data/${name}
#     done

# fi

# if [ $stage -le 5 ];then 
#   utils/combine_data.sh --extra-files "utt2num_frames" data/dihard2_train data/dihard2_train_dev data/dihard2_train_eval
#   utils/fix_data_dir.sh data/dihard2_train
# fi


