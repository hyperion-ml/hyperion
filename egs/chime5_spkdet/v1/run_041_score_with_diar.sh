#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

net_name=3b
diar_name=diar3b_t-0.9

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
score_plda_dir=$score_dir/plda_${diar_name}


if [ $stage -le 1 ]; then

    echo "Chime5 with diarization ${diar_name}"
    steps_be/eval_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			   data/chime5_spkdet_test/trials data/chime5_spkdet_test_${diar_name}/trials \
			   data/chime5_spkdet_enroll/utt2spk \
			   $xvector_dir/chime5_spkdet_${diar_name}/xvector.scp \
			   data/chime5_spkdet_test_${diar_name}/utt2orig \
			   $be_dir/lda_lnorm.h5 \
			   $be_dir/plda.h5 \
			   $score_plda_dir/chime5_spkdet_scores 
    
    local/score_chime5_spkdet.sh data/chime5_spkdet_test $score_plda_dir &

fi

if [ $stage -le 2 ];then
    local/calibrate_chime5_spkdet_v1.sh --cmd "$train_cmd" $score_plda_dir 
    local/score_chime5_spkdet.sh data/chime5_spkdet_test ${score_plda_dir}_cal_v1 
    
fi

    
exit
