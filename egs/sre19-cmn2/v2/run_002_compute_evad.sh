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
vad_config=conf/vad_8k.yaml

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $vaddir/storage ]; then
	dir_name=$USER/hyp-data/sre19-cmn2-v2/$storage_name/vad/storage
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

#Train datasets
if [ $stage -le 2 ];then
    for name in sre_tel swbd voxcelebcat_tel \
    		        sre18_cmn2_adapt_lab sre18_dev_unlabeled \
    			sre18_eval40_enroll_cmn2 sre18_eval40_test_cmn2 \
    			sre19_eval_enroll_cmn2 sre19_eval_test_cmn2
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 40 ? $num_spk:40))
	hyp_utils/feats/make_evad.sh --write-utt2num-frames true \
	    --vad-config $vad_config --nj $nj --cmd "$train_cmd" \
	    data/${name} exp/make_vad/$name $vaddir
	utils/fix_data_dir.sh data/${name}
    done
fi

exit

