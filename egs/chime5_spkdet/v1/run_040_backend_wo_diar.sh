#!/bin/bash
# Copyright       2019   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

net_name=3b

lda_dim=300
plda_y_dim=175
plda_z_dim=200

stage=1

. parse_options.sh || exit 1;


xvector_dir=exp/xvectors/$net_name

plda_data=train_combined
plda_type=splda
plda_label=${plda_type}y${plda_y_dim}_v1

be_name=lda${lda_dim}_${plda_label}_${plda_data}
be_dir=exp/be/$net_name/$be_name

score_dir=exp/scores/$net_name/${be_name}
score_plda_dir=$score_dir/plda
score_plda_gtvad_dir=$score_dir/plda_gtvad

#train_cmd=run.pl

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
    exit
fi

if [ $stage -le 3 ];then
    local/calibrate_chime5_spkdet_v1.sh --cmd "$train_cmd" $score_plda_dir &
    local/calibrate_chime5_spkdet_v1.sh --cmd "$train_cmd" $score_plda_gtvad_dir &
    wait

    
    local/score_chime5_spkdet.sh data/chime5_spkdet_test ${score_plda_dir}_cal_v1 &
    local/score_chime5_spkdet.sh data/chime5_spkdet_test ${score_plda_gtvad_dir}_cal_v1  &
    wait
    exit
fi

