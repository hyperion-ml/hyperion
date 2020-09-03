#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

#spk det back-end
lda_dim=150
ncoh=400
plda_y_dim=125
plda_z_dim=150

plda_type=splda
#plda_data=sre_tel
coh_data=sre18_dev_unlabeled

ft=0

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
fi

plda_label=${plda_type}y${plda_y_dim}
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda

if [ ! -d scoring_software/sre19-cmn2 ];then
    local/download_sre19cmn2_scoring_tool.sh 
fi

if [ $stage -le 1 ]; then

    steps_be/train_tel_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_dim \
    				--plda_type $plda_type \
    				--y_dim $plda_y_dim --z_dim $plda_z_dim \
    				$xvector_dir/$plda_data/xvector.scp \
    				data/$plda_data \
    				$be_dir 

    

fi

if [ $stage -le 2 ]; then

    #SRE16
    echo "eval SRE16 without S-Norm"

    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			       data/sre16_eval40_yue_test/trials \
			       data/sre16_eval40_yue_enroll/utt2spk \
			       $xvector_dir/sre16_eval40_yue/xvector.scp \
			       $be_dir/lda_lnorm.h5 \
			       $be_dir/plda.h5 \
			       $score_plda_dir/sre16_eval40_yue_scores &

    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			       data/sre16_eval40_tgl_test/trials \
			       data/sre16_eval40_tgl_enroll/utt2spk \
			       $xvector_dir/sre16_eval40_tgl/xvector.scp \
			       $be_dir/lda_lnorm.h5 \
			       $be_dir/plda.h5 \
			       $score_plda_dir/sre16_eval40_tgl_scores &

    echo "eval SRE19 without S-Norm"
    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			       data/sre19_eval_test_cmn2/trials \
			       data/sre19_eval_enroll_cmn2/utt2spk \
			       $xvector_dir/sre19_eval_cmn2/xvector.scp \
			       $be_dir/lda_lnorm.h5 \
			       $be_dir/plda.h5 \
			       $score_plda_dir/sre19_eval_cmn2_scores &

    echo "eval SRE20 without S-Norm"
    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			       data/sre20cts_eval_test/trials \
			       data/sre20cts_eval_enroll/utt2spk \
			       $xvector_dir/sre19cts_eval/xvector.scp \
			       $be_dir/lda_lnorm.h5 \
			       $be_dir/plda.h5 \
			       $score_plda_dir/sre19cts_eval_scores &

    
    wait
    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $score_plda_dir
    local/make_sre20cts_sub.sh $sre20cts_eval_root ${score_plda_dir}/sre20cts_eval_scores
fi
exit

if [ $stage -le 3 ];then
    local/calibrate_sre19cmn2_v1.sh --cmd "$train_cmd" eval40 $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_eval40_test_cmn2 eval40 ${score_plda_dir}_cal_v1eval40
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1eval40

    #local/make_sre19cmn2_sub.sh $sre19cmn2_eval_root ${score_plda_dir}_cal_v1dev/sre19_eval_cmn2_scores
fi

score_plda_dir=$score_dir/plda_snorm

if [ $stage -le 4 ]; then

    #SRE18
    echo "SRE18 S-Norm"
    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
				     data/sre18_eval40_test_cmn2/trials \
				     data/sre18_eval40_enroll_cmn2/utt2spk \
				     $xvector_dir/sre18_eval40_cmn2/xvector.scp \
				     data/${coh_data}/utt2spk \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/lda_lnorm_adapt.h5 \
				     $be_dir/plda_adapt2.h5 \
				     $score_plda_dir/sre18_eval40_cmn2_scores &

    echo "SRE19 S-Norm"
    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
				     data/sre19_eval_test_cmn2/trials \
				     data/sre19_eval_enroll_cmn2/utt2spk \
				     $xvector_dir/sre19_eval_cmn2/xvector.scp \
				     data/${coh_data}/utt2spk \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/lda_lnorm_adapt.h5 \
				     $be_dir/plda_adapt2.h5 \
				     $score_plda_dir/sre19_eval_cmn2_scores &

    wait
    local/score_sre18cmn2.sh data/sre18_eval40_test_cmn2 eval40 $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $score_plda_dir
fi

if [ $stage -le 5 ];then
    local/calibrate_sre19cmn2_v1.sh --cmd "$train_cmd" eval40 $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_eval40_test_cmn2 eval40 ${score_plda_dir}_cal_v1eval40
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1eval40
fi

    
exit
