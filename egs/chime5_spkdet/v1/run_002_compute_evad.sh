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
vaddir_gt=`pwd`/exp/vad_gt
vad_config=conf/vad_16k.yaml

stage=1
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file


if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $vaddir/storage ]; then
	dir_name=$USER/hyp-data/sre19-av-a/v2.1/$storage_name/vad/storage
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

# Train/Test Datasets
if [ $stage -le 2 ];then 
    for name in chime5_spkdet_test voxcelebcat
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 40 ? $num_spk:40))
	hyp_utils/feats/make_evad.sh --write-utt2num-frames true \
	    --vad-config $vad_config --nj $nj --cmd "$train_cmd" \
	    data/${name} exp/make_vad/$name $vaddir
	utils/fix_data_dir.sh data/${name}
    done
fi

if [ $stage -le 3 ];then
    for name in chime5_spkdet_enroll
    do
	hyp_utils/feats/segments_to_bin_vad.sh --nj 5 data/$name/diarization_segments data/$name $vaddir_gt
	utils/fix_data_dir.sh data/$name
    done
    for name in chime5_spkdet_test
    do
	rm -rf data/${name}_gtvad
	cp -r data/$name data/${name}_gtvad
	name=${name}_gtvad
	hyp_utils/feats/segments_to_bin_vad.sh --nj 5 data/$name/diarization_segments data/$name $vaddir_gt
	utils/fix_data_dir.sh data/$name
    done
fi



