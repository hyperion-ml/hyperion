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

plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda
score_plda_gtvad_dir=$score_dir/plda_gtvad


if [ $stage -le 1 ]; then
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
	--lda_dim $lda_dim \
	--plda_type $plda_type \
	--y_dim $plda_y_dim --z_dim $plda_z_dim \
	$xvector_dir/$plda_data/xvector.scp \
	data/$plda_data \
	$be_dir 
fi

if [ $stage -le 2 ];then

    echo "Chime5 wo diarization"
    steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			   data/chime5_spkdet_test/trials \
			   data/chime5_spkdet_enroll/utt2spk \
			   $xvector_dir/chime5_spkdet/xvector.scp \
			   $be_dir/lda_lnorm.h5 \
			   $be_dir/plda.h5 \
			   $score_plda_dir/chime5_spkdet_scores &


    echo "Chime5 with ground-truth diarization"
    steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			   data/chime5_spkdet_test/trials \
			   data/chime5_spkdet_enroll/utt2spk \
			   $xvector_dir/chime5_spkdet_gtvad/xvector.scp \
			   $be_dir/lda_lnorm.h5 \
			   $be_dir/plda.h5 \
			   $score_plda_gtvad_dir/chime5_spkdet_scores &

    wait

    local/score_chime5_spkdet.sh data/chime5_spkdet_test $score_plda_dir &
    local/score_chime5_spkdet.sh data/chime5_spkdet_test $score_plda_gtvad_dir &
    wait
fi

if [ $stage -le 3 ];then
    local/calibrate_chime5_spkdet_v1.sh --cmd "$train_cmd" $score_plda_dir &
    local/calibrate_chime5_spkdet_v1.sh --cmd "$train_cmd" $score_plda_gtvad_dir &
    wait

    
    local/score_chime5_spkdet.sh data/chime5_spkdet_test ${score_plda_dir}_cal_v1 &
    local/score_chime5_spkdet.sh data/chime5_spkdet_test ${score_plda_gtvad_dir}_cal_v1  &
    wait
fi


