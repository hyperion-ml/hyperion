#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
use_gpu=false
xvec_chunk_length=12800
ft=1
. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
    nnet=$ft_nnet
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
    nnet=$ft2_nnet
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
    nnet=$ft3_nnet
fi

xvector_dir=exp/xvectors/$nnet_name

if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in sre_tel fisher_spa \
			# cncelebcat_tel 
    			# $cv_noeng_datasets $babel_datasets $lre17_datasets
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	if [ $plda_num_augs -eq 0 ];then
	    nj=$(($num_spk < 100 ? $num_spk:100))
    	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj $nj ${xvec_args} \
		--random-utt-length true --min-utt-length 1000 --max-utt-length 6000 \
		--feat-config $feat_config \
    		$nnet data/${name} \
    		$xvector_dir/${name}
	else
	    nj=$(($num_spk < 300 ? $num_spk:300))
	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj $nj ${xvec_args} \
		--random-utt-length true --min-utt-length 1000 --max-utt-length 6000 \
		--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    		$nnet data/${name} \
    		$xvector_dir/${name}_augx${plda_num_augs} \
		data/${name}_augx${plda_num_augs}
	fi
    done
fi

if [ $stage -le 2 ]; then
    # Extract xvectors for adapting LDA/PLDA
    for name in sre16_train_dev_cmn \
		    sre16_train_dev_ceb \
    		    sre16_eval_tr60_yue sre16_eval_tr60_tgl \
		    sre18_cmn2_train_lab sre18_dev_unlabeled 
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	if [ $plda_num_augs -eq 0 ];then
	    nj=$(($num_spk < 30 ? $num_spk:30))
    	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj $nj ${xvec_args} \
		--feat-config $feat_config \
    		$nnet data/${name} \
    		$xvector_dir/${name}
	else
	    nj=$(($num_spk < 100 ? $num_spk:100))
	    steps_xvec/extract_xvectors_from_wav.sh \
		--cmd "$xvec_cmd --mem 12G" --nj $nj ${xvec_args} \
		--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    		$nnet data/${name} \
    		$xvector_dir/${name}_augx${plda_num_augs} \
		data/${name}_augx${plda_num_augs}
	fi
    done
fi


if [ $stage -le 3 ]; then
    # Extracts x-vectors for evaluation
    for name in sre16_eval40_yue_enroll \
		    sre16_eval40_yue_test \
    		    sre16_eval40_tgl_enroll sre16_eval40_tgl_test \
		    sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 \
    		    sre20cts_eval_enroll sre20cts_eval_test
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/$name
    done
fi


if [ $stage -le 4 ]; then
    # merge eval x-vectors lists
    mkdir -p $xvector_dir/sre16_eval40_yue
    cat $xvector_dir/sre16_eval40_yue_{enroll,test}/xvector.scp > $xvector_dir/sre16_eval40_yue/xvector.scp
    mkdir -p $xvector_dir/sre16_eval40_tgl
    cat $xvector_dir/sre16_eval40_tgl_{enroll,test}/xvector.scp > $xvector_dir/sre16_eval40_tgl/xvector.scp
    mkdir -p $xvector_dir/sre19_eval_cmn2
    cat $xvector_dir/sre19_eval_{enroll,test}_cmn2/xvector.scp > $xvector_dir/sre19_eval_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre20cts_eval
    cat $xvector_dir/sre20cts_eval_{enroll,test}/xvector.scp > $xvector_dir/sre20cts_eval/xvector.scp
fi

# if [ $stage -le 5 ];then
#     # this is to avoid "Too many open files error" when reading the x-vectors in the back-end
#     # we need to combine the ark files into a single ark file
#     for name in $cv_noeng_datasets $babel_datasets $lre17_datasets
#     do
# 	mv $xvector_dir/$name/xvector.scp $xvector_dir/$name/xvector.tmp.scp 
# 	copy-vector scp:$xvector_dir/$name/xvector.tmp.scp ark,scp:$xvector_dir/$name/xvector.ark,$xvector_dir/$name/xvector.scp
#     done
# fi

if [ $stage -le 6 ];then
    # merge datasets and x-vector list for plda training
    # utils/combine_data.sh --extra-files "utt2dur utt2num_frames utt2lang" \
    # 			  data/alleng \
    # 			  data/sre_tel data/voxcelebcat_tel data/cvcat_en_tel

    utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    			  data/sre16-8 \
    			  data/sre16_train_dev_cmn data/sre16_train_dev_ceb \
    			  data/sre16_eval_tr60_yue data/sre16_eval_tr60_tgl \
    			  data/sre18_cmn2_train_lab
    mkdir -p $xvector_dir/sre16-8
    cat $xvector_dir/{sre16_train_dev_cmn,sre16_train_dev_ceb,sre16_eval_tr60_yue,sre16_eval_tr60_tgl,sre18_cmn2_train_lab}/xvector.scp \
    	> $xvector_dir/sre16-8/xvector.scp
    

    utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    			  data/realtel_noeng \
    			  data/sre16-8 data/fisher_spa
    mkdir -p $xvector_dir/realtel_noeng
    cat $xvector_dir/{sre16-8,fisher_spa}/xvector.scp > $xvector_dir/realtel_noeng/xvector.scp

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/sre16zh-8 \
    # 			  data/sre16_train_dev_cmn \
    # 			  data/sre16_eval_tr60_yue \
    # 			  data/sre18_cmn2_train_lab
    # mkdir -p $xvector_dir/sre16zh-8
    # cat $xvector_dir/{sre16_train_dev_cmn,sre16_eval_tr60_yue,sre18_cmn2_train_lab}/xvector.scp \
    # 	> $xvector_dir/sre16zh-8/xvector.scp
    

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/sre16zh-8_fisher_spa \
    # 			  data/sre16zh-8 data/fisher_spa
    # mkdir -p $xvector_dir/sre16zh-8_fisher_spa
    # cat $xvector_dir/{sre16zh-8,fisher_spa}/xvector.scp > $xvector_dir/sre16zh-8_fisher_spa/xvector.scp


    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/cvcat_noeng_tel \
    # 			  $(echo $cv_noeng_datasets | sed 's@cvcat_@data/cvcat_@g')
    # mkdir -p $xvector_dir/cvcat_noeng_tel
    # cat $(echo $cv_noeng_datasets | sed -e 's@cvcat_\([^_]*\)_tel@'$xvector_dir'/cvcat_\1_tel/xvector.scp@g' ) > $xvector_dir/cvcat_noeng_tel/xvector.scp  

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/allnoeng \
    # 			  data/sre16-8 \
    # 			  data/cvcat_noeng_tel \
    # 			  data/cncelebcat_tel data/fisher_spa
    # mkdir -p $xvector_dir/allnoeng
    # cat $xvector_dir/{sre16-8,cvcat_noeng_tel,cncelebcat_tel,fisher_spa}/xvector.scp > $xvector_dir/allnoeng/xvector.scp

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/alllangs \
    # 			  data/sre_tel \
    # 			  data/allnoeng
    # mkdir -p $xvector_dir/alllangs
    # cat $xvector_dir/{sre_tel,allnoeng}/xvector.scp > $xvector_dir/alllangs/xvector.scp
  
    utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    			  data/realtel_alllangs data/sre_tel data/realtel_noeng
    mkdir -p $xvector_dir/realtel_alllangs
    cat $xvector_dir/{sre_tel,realtel_noeng}/xvector.scp > $xvector_dir/realtel_alllangs/xvector.scp

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/sre16-8_cncelebcat_tel data/sre16-8 data/cncelebcat_tel
    # mkdir -p $xvector_dir/sre16-8_cncelebcat_tel
    # cat $xvector_dir/{sre16-8,cncelebcat_tel}/xvector.scp > $xvector_dir/sre16-8_cncelebcat_tel/xvector.scp

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/sre16-8_cvcat_zh-HK data/sre16-8 data/cvcat_zh-HK_tel
    # mkdir -p $xvector_dir/sre16-8_cvcat_zh-HK
    # cat $xvector_dir/{sre16-8,cvcat_zh-HK_tel}/xvector.scp > $xvector_dir/sre16-8_cvcat_zh-HK/xvector.scp
    
    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/sre16-8_cvcat_zh data/sre16-8 data/cvcat_zh-HK_tel data/cvcat_zh-CN_tel data/cvcat_zh-TW_tel
    # mkdir -p $xvector_dir/sre16-8_cvcat_zh
    # cat $xvector_dir/{sre16-8,cvcat_zh-HK_tel,cvcat_zh-CN_tel,cvcat_zh-TW_tel}/xvector.scp > $xvector_dir/sre16-8_cvcat_zh/xvector.scp

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 		      data/sre16-8_cvcat_ar data/sre16-8 data/cvcat_ar_tel
    # mkdir -p $xvector_dir/sre16-8_cvcat_ar
    # cat $xvector_dir/{sre16-8,cvcat_ar_tel}/xvector.scp > $xvector_dir/sre16-8_cvcat_ar/xvector.scp

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/cvcat_zh data/cvcat_zh-HK_tel data/cvcat_zh-CN_tel data/cvcat_zh-TW_tel
    # mkdir -p $xvector_dir/cvcat_zh
    # cat $xvector_dir/{cvcat_zh-HK_tel,cvcat_zh-CN_tel,cvcat_zh-TW_tel}/xvector.scp > $xvector_dir/cvcat_zh/xvector.scp

    
    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/babel_alllangs \
    # 			  $(echo $babel_datasets | sed 's@babel@data/babel@g')
    # mkdir -p $xvector_dir/babel_alllangs
    # cat $(echo $babel_datasets | sed -e 's@\([^ ]*\)@'$xvector_dir'/\1/xvector.scp@g' ) > $xvector_dir/babel_alllangs/xvector.scp  


    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/lre17_alllangs \
    # 			  $(echo $lre17_datasets | sed 's@lre17@data/lre17@g')
    # mkdir -p $xvector_dir/lre17_alllangs
    # cat $(echo $lre17_datasets | sed -e 's@\([^ ]*\)@'$xvector_dir'/\1/xvector.scp@g' ) > $xvector_dir/lre17_alllangs/xvector.scp  

    # utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    # 			  data/realtel_alllangs_labunlab \
    # 			  data/realtel_alllangs data/babel_alllangs data/sre18_dev_unlabeled
    # mkdir -p $xvector_dir/realtel_alllangs_labunlab
    # cat $xvector_dir/{realtel_alllangs,babel_alllangs,sre18_dev_unlabeled}/xvector.scp \
    # 	> $xvector_dir/realtel_alllangs_labunlab/xvector.scp

fi
