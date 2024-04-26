#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e
nodes=b1
storage_name=$(date +'%m_%d_%H_%M')
vaddir=`pwd`/exp/vad_e

stage=1
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file


if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $vaddir/storage ]; then
	dir_name=$USER/hyp-data/lre22-fixed-v1.8k-$storage_name/vad/storage
	if [ "$nodes" == "b0" ];then
	    utils/create_split_dir.pl \
			    utils/create_split_dir.pl \
		/export/b{04,05,06,07}/$dir_name $vaddir/storage
	elif [ "$nodes" == "b1" ];then
	    utils/create_split_dir.pl \
		/export/b1{0,1,2,3,4,5,6,7,8,9}/$dir_name $vaddir/storage
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
  for name in voxlingua107 \
		lre17_train \
		lre17_dev_cts lre17_dev_afv \
		lre17_eval_cts lre17_eval_afv \
		lre22_dev lre22_eval \
  do
    num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
    nj=$(($num_spk < 40 ? $num_spk:40))
    hyp_utils/feats/make_evad.sh --write-utt2num-frames true \
				 --vad-config $vad_config --nj $nj --cmd "$train_cmd" \
				 data/${name} exp/make_vad/$name $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

