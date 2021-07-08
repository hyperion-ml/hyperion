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

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 


plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda_${diar_name}


if [ $stage -le 1 ]; then

    echo "Chime5 with diarization ${diar_name}"
    steps_be/eval_be_diar_v2.sh --cmd "$train_cmd" --plda_type $plda_type \
			   data/chime5_spkdet_test/trials \
			   data/chime5_spkdet_enroll/utt2spk \
			   $xvector_dir/chime5_spkdet_enroll/xvector.scp \
			   $xvector_dir/chime5_spkdet_test_${diar_name}/xvector.scp \
			   $be_dir/lda_lnorm.h5 \
			   $be_dir/plda.h5 \
			   $score_plda_dir/chime5_spkdet_scores 
    
    local/score_chime5_spkdet.sh data/chime5_spkdet_test $score_plda_dir 

fi

if [ $stage -le 2 ];then
    local/calibrate_chime5_spkdet_v1.sh --cmd "$train_cmd" $score_plda_dir 
    local/score_chime5_spkdet.sh data/chime5_spkdet_test ${score_plda_dir}_cal_v1 
fi

    
exit
