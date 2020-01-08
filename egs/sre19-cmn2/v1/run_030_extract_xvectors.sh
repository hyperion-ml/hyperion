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

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ];then
    # remove reverb from xxx_combined datasets, we will use for train plda later
    for name in sre_tel sre18_train_eval_cmn2
    do
	if [ ! -d data/${name}_combined_noreverb ];then
	    cp -r data/${name}_combined data/${name}_combined_noreverb
	    awk '$1 !~ /reverb/' data/${name}_combined/utt2spk > data/${name}_combined_noreverb/utt2spk
	    utils/fix_data_dir.sh data/${name}_combined_noreverb
	fi
    done
    
fi

if [ $stage -le 2 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in  sre_tel sre18_train_eval_cmn2 #sre_phnmic voxceleb
    do
	name_c=${name}_combined_noreverb
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 180 \
					     $nnet_dir data/${name_c} \
					     $xvector_dir/${name_c}

	#create a xvector dir for clean only xvectors
	mkdir -p $xvector_dir/${name}
	cp $xvector_dir/${name_c}/xvector.scp $xvector_dir/${name}
    done
fi

if [ $stage -le 3 ]; then
    # Extracts x-vectors for evaluation
    for name in  sre18_train_dev_cmn2 \
	sre18_dev_unlabeled sre18_dev_enroll_cmn2 sre18_dev_test_cmn2 \
    	sre18_eval_enroll_cmn2 sre18_eval_test_cmn2 \
    	sre19_eval_enroll_cmn2 sre19_eval_test_cmn2
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 80 ? $num_spk:80))
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done
fi

if [ $stage -le 4 ]; then
    mkdir -p $xvector_dir/sre18_dev_cmn2
    cat $xvector_dir/sre18_dev_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_dev_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre18_eval_cmn2
    cat $xvector_dir/sre18_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_eval_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre19_eval_cmn2
    cat $xvector_dir/sre19_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre19_eval_cmn2/xvector.scp
fi


exit
