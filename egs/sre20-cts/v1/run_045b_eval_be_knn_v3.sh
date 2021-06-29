#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
# Back-end with two-stage kNN
# Calibration depend on # of enrollment cuts
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

lda_dim=150
ncoh=500
plda_y_dim=125
plda_z_dim=150
knn1=800
knn2=300
w_mu=1
w_B=0.5
w_W=0.5
pca_var_r=0.975

plda_type=splda
plda_data=realtel_alllangs
coh_data=realtel_alllangs
# cal_set=sre16-9
cal_set=sre16-yue
ft=1

. parse_options.sh || exit 1;
. $config_file
. datapath.sh
# plda_data=sre16-8
# plda_data=realtel_noeng
# plda_data=cvcat_noeng_tel
# plda_data=allnoeng
# plda_data=alllangs
# plda_data=cncelebcat_tel
# plda_data=realtel_alllangs
# plda_data=sre16-8_cncelebcat_tel
# plda_data=sre16-8_cvcat_zh-HKs
# plda_data=sre16-8_cvcat_zh
# plda_data=sre16-8_cvcat_ar
# plda_data=cvcat_zh

if [ $ft -eq 1 ];then
    nnet_name=$ft_nnet_name
elif [ $ft -eq 2 ];then
    nnet_name=$ft2_nnet_name
elif [ $ft -eq 3 ];then
    nnet_name=$ft3_nnet_name
fi

plda_label=${plda_type}y${plda_y_dim}
be_name=pca${pca_var_r}_knn${knn1}-${knn2}_lda${lda_dim}_${plda_label}_${plda_data}_v3_adapt_mu${w_mu}B${w_B}W${w_W}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda

if [ ! -d scoring_software/sre19-cmn2 ];then
    local/download_sre19cmn2_scoring_tool.sh 
fi

if [ $stage -le 1 ]; then

    for name in sre16_eval40_tgl_enroll sre16_eval40_yue_enroll \
					sre19_eval_enroll_cmn2 sre20cts_eval_enroll
    do
	echo "Train BE for $name"
	steps_be/train_tel_be_knn_v3.sh --cmd "$train_cmd" --nj 40 --knn1 $knn1 --knn2 $knn2 --pca-var-r $pca_var_r \
    					--lda-dim $lda_dim \
    					--plda-type $plda_type \
    					--y-dim $plda_y_dim --z-dim $plda_z_dim \
					--w-mu $w_mu --w-B $w_B --w-W $w_W \
    					$xvector_dir/$plda_data/xvector.scp \
    					data/$plda_data \
					$xvector_dir/$name/xvector.scp \
					data/$name \
    					$be_dir/$name &
    done
    wait
fi

if [ $stage -le 2 ]; then

    #SRE16
    echo "eval SRE16 without S-Norm"

    steps_be/eval_tel_be_knn_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			       data/sre16_eval40_yue_test/trials \
			       data/sre16_eval40_yue_enroll/utt2spk \
			       $xvector_dir/sre16_eval40_yue/xvector.scp \
			       $be_dir/sre16_eval40_yue_enroll \
			       $score_plda_dir/sre16_eval40_yue_scores &

    steps_be/eval_tel_be_knn_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
				   data/sre16_eval40_tgl_test/trials \
				   data/sre16_eval40_tgl_enroll/utt2spk \
				   $xvector_dir/sre16_eval40_tgl/xvector.scp \
				   $be_dir/sre16_eval40_tgl_enroll \
				   $score_plda_dir/sre16_eval40_tgl_scores &

    echo "eval SRE19 without S-Norm"
    steps_be/eval_tel_be_knn_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			       data/sre19_eval_test_cmn2/trials \
			       data/sre19_eval_enroll_cmn2/utt2spk \
			       $xvector_dir/sre19_eval_cmn2/xvector.scp \
			       $be_dir/sre19_eval_enroll_cmn2 \
			       $score_plda_dir/sre19_eval_cmn2_scores &

    echo "eval SRE20 without S-Norm"
    steps_be/eval_tel_be_knn_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
    			       data/sre20cts_eval_test/trials \
    			       data/sre20cts_eval_enroll/utt2spk \
    			       $xvector_dir/sre20cts_eval/xvector.scp \
    			       $be_dir/sre20cts_eval_enroll \
    			       $score_plda_dir/sre20cts_eval_scores &

    
    wait
    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $score_plda_dir
    #local/make_sre20cts_sub.sh $sre20cts_eval_root ${score_plda_dir}/sre20cts_eval_scores
fi


if [ $stage -le 3 ];then
    local/calibrate_sre20cts_v2.sh --cmd "$train_cmd" $cal_set $score_plda_dir
    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue ${score_plda_dir}_cal_v2${cal_set}
    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl ${score_plda_dir}_cal_v2${cal_set}
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v2${cal_set}
fi

exit

########################################################################

score_plda_dir=$score_dir/plda_snorm_${coh_data}${ncoh}

if [ $stage -le 4 ]; then

    echo "eval SRE16 with S-Norm"
    steps_be/eval_tel_be_knn_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
				     data/sre16_eval40_yue_test/trials \
				     data/sre16_eval40_yue_enroll/utt2spk \
				     $xvector_dir/sre16_eval40_yue/xvector.scp \
				     data/${coh_data}/utt2spk \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre16_eval40_yue_enroll \
				     $score_plda_dir/sre16_eval40_yue_scores &
    
    steps_be/eval_tel_be_knn_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type  --ncoh $ncoh \
				     data/sre16_eval40_tgl_test/trials \
				     data/sre16_eval40_tgl_enroll/utt2spk \
				     $xvector_dir/sre16_eval40_tgl/xvector.scp \
				     data/${coh_data}/utt2spk \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre16_eval40_tgl_enroll \
				     $score_plda_dir/sre16_eval40_tgl_scores &

    echo "eval SRE19 with S-Norm"
    steps_be/eval_tel_be_knn_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type  --ncoh $ncoh \
				     data/sre19_eval_test_cmn2/trials \
				     data/sre19_eval_enroll_cmn2/utt2spk \
				     $xvector_dir/sre19_eval_cmn2/xvector.scp \
				     data/${coh_data}/utt2spk \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre19_eval_enroll_cmn2 \
				     $score_plda_dir/sre19_eval_cmn2_scores &

    echo "eval SRE20 with S-Norm"
    steps_be/eval_tel_be_knn_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type  --ncoh $ncoh \
    				     data/sre20cts_eval_test/trials \
    				     data/sre20cts_eval_enroll/utt2spk \
    				     $xvector_dir/sre20cts_eval/xvector.scp \
				     data/${coh_data}/utt2spk \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre20cts_eval_enroll \
    				     $score_plda_dir/sre20cts_eval_scores &
    
    wait
    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $score_plda_dir
    
fi

if [ $stage -le 5 ];then
    local/calibrate_sre20cts_v1.sh --cmd "$train_cmd" $cal_set $score_plda_dir
    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue ${score_plda_dir}_cal_v1${cal_set}
    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl ${score_plda_dir}_cal_v1${cal_set}
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1${cal_set}
    exit
fi

score_plda_dir=$score_dir/plda_snorm_v1.1_${coh_data}${ncoh}

if [ $stage -le 6 ]; then

    echo "eval SRE16 with S-Norm"
    steps_be/eval_tel_be_knn_snorm_v1.1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
				     data/sre16_eval40_yue_test/trials \
				     data/sre16_eval40_yue_enroll/utt2spk \
				     $xvector_dir/sre16_eval40_yue/xvector.scp \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre16_eval40_yue_enroll \
				     $score_plda_dir/sre16_eval40_yue_scores &
    
    steps_be/eval_tel_be_knn_snorm_v1.1.sh --cmd "$train_cmd" --plda_type $plda_type  --ncoh $ncoh \
				     data/sre16_eval40_tgl_test/trials \
				     data/sre16_eval40_tgl_enroll/utt2spk \
				     $xvector_dir/sre16_eval40_tgl/xvector.scp \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre16_eval40_tgl_enroll \
				     $score_plda_dir/sre16_eval40_tgl_scores &

    echo "eval SRE19 with S-Norm"
    steps_be/eval_tel_be_knn_snorm_v1.1.sh --cmd "$train_cmd" --plda_type $plda_type  --ncoh $ncoh \
				     data/sre19_eval_test_cmn2/trials \
				     data/sre19_eval_enroll_cmn2/utt2spk \
				     $xvector_dir/sre19_eval_cmn2/xvector.scp \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre19_eval_enroll_cmn2 \
				     $score_plda_dir/sre19_eval_cmn2_scores &

    echo "eval SRE20 with S-Norm"
    steps_be/eval_tel_be_knn_snorm_v1.1.sh --cmd "$train_cmd" --plda_type $plda_type  --ncoh $ncoh \
    				     data/sre20cts_eval_test/trials \
    				     data/sre20cts_eval_enroll/utt2spk \
    				     $xvector_dir/sre20cts_eval/xvector.scp \
				     $xvector_dir/${coh_data}/xvector.scp \
				     $be_dir/sre20cts_eval_enroll \
    				     $score_plda_dir/sre20cts_eval_scores &
    
    wait
    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue $score_plda_dir
    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 $score_plda_dir
    
fi

if [ $stage -le 7 ];then
    local/calibrate_sre20cts_v1.sh --cmd "$train_cmd" $cal_set $score_plda_dir
    local/score_sre16.sh data/sre16_eval40_yue_test eval40_yue ${score_plda_dir}_cal_v1${cal_set}
    local/score_sre16.sh data/sre16_eval40_tgl_test eval40_tgl ${score_plda_dir}_cal_v1${cal_set}
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1${cal_set}
fi

    
exit
