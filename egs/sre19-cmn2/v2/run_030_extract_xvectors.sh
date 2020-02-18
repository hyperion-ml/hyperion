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
use_gpu=false

. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length 12800"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ];then
    # remove reverb from xxx_combined datasets, we will use for train plda later
    for name in sre_tel sre18_cmn2_adapt_lab
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
    for name in sre_tel
    do
    	steps_xvec/extract_xvectors.sh --cmd "$xvec_cmd --mem 12G" --nj 300 ${xvec_args} \
	    --random-utt-length true --min-utt-length 100 --max-utt-length 6000 \
    	    $nnet data/${name}_combined_noreverb \
    	    $xvector_dir/${name}_combined_noreverb
	mkdir -p $xvector_dir/${name}
	cp $xvector_dir/${name}_combined_noreverb/xvector.scp $xvector_dir/${name}
    done

    for name in sre18_cmn2_adapt_lab
    do
    	steps_xvec/extract_xvectors.sh --cmd "$xvec_cmd --mem 12G" --nj 100 ${xvec_args} \
    	    $nnet data/${name}_combined_noreverb \
    	    $xvector_dir/${name}_combined_noreverb
	mkdir -p $xvector_dir/${name}
	cp $xvector_dir/${name}_combined_noreverb/xvector.scp $xvector_dir/${name}
    done
    
fi

if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in sre18_dev_unlabeled \
		    sre18_eval40_enroll_cmn2 sre18_eval40_test_cmn2 \
		    sre19_eval_enroll_cmn2 sre19_eval_test_cmn2
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
					      $nnet data/$name \
					      $xvector_dir/$name
    done
fi


if [ $stage -le 4 ]; then
    mkdir -p $xvector_dir/sre18_eval40_cmn2
    cat $xvector_dir/sre18_eval40_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre18_eval40_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre19_eval_cmn2
    cat $xvector_dir/sre19_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre19_eval_cmn2/xvector.scp
fi


exit
