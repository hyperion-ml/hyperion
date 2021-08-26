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

coh_data=janus_dev_test_core
ncoh=1000

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

#plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v1
#be_vid_name=lda${lda_vid_dim}_${plda_vid_label}_${plda_vid_data}

be_vid_name=be_v1

face_embed_ref_dir=$face_embed_dir/ref
face_embed_facedet_dir=$face_embed_dir/facedet

# face_embed_name=1.5
# face_embed_dir=`pwd`/exp/face_embed
# face_embed_ref_dir=$face_embed_dir/v1/ref
# face_embed_facedet_dir=$face_embed_dir/v2/facedet

# face_embed_name=v2
# face_embed_dir=`pwd`/exp/face_embed
# face_embed_ref_dir=$face_embed_dir/v2/ref
# face_embed_facedet_dir=$face_embed_dir/v2/facedet

be_vid_dir=exp/be/$face_embed_name/$be_vid_name
score_dir=exp/scores/$face_embed_name/${be_vid_name}
score_plda_dir=$score_dir/cosine



if [ $stage -le 4 ]; then

    echo "SRE19"

    steps_be/eval_face_vid_be_v1.sh --cmd "$train_cmd" \
    	data/sre19_av_v_dev_test/trials \
    	data/sre19_av_v_dev_enroll/utt2spk \
    	$face_embed_ref_dir/sre19_av_v_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_dev_test/embed.scp \
    	$score_plda_dir/sre19_av_v_dev_scores &

    
    steps_be/eval_face_vid_be_v1.sh --cmd "$train_cmd" \
    	data/sre19_av_v_eval_test/trials \
    	data/sre19_av_v_eval_enroll/utt2spk \
    	$face_embed_ref_dir/sre19_av_v_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_eval_test/embed.scp \
    	$score_plda_dir/sre19_av_v_eval_scores &

    wait

    local/score_sre19av.sh data/sre19_av_v_dev_test v_dev $score_plda_dir
    local/score_sre19av.sh data/sre19_av_v_eval_test v_eval $score_plda_dir

fi


if [ $stage -le 5 ]; then

    # JANUS
    echo "JANUS core"

    steps_be/eval_face_vid_be_v1.sh --cmd "$train_cmd" \
    	data/janus_dev_test_core/trials \
    	data/janus_dev_enroll/utt2spk \
    	$face_embed_ref_dir/janus_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_dev_test_core/embed.scp \
    	$score_plda_dir/janus_dev_core_scores &

    
    steps_be/eval_face_vid_be_v1.sh --cmd "$train_cmd" \
    	data/janus_eval_test_core/trials \
    	data/janus_eval_enroll/utt2spk \
    	$face_embed_ref_dir/janus_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_eval_test_core/embed.scp \
    	$score_plda_dir/janus_eval_core_scores &

    wait

    local/score_janus_core.sh data/janus_dev_test_core dev $score_plda_dir
    local/score_janus_core.sh data/janus_eval_test_core eval $score_plda_dir
fi




if [ $stage -le 6 ];then

    local/calibrate_sre19av_v_v1_sre19.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sre19av.sh data/sre19_av_v_dev_test v_dev ${score_plda_dir}_cal_v1_sre19
    local/score_sre19av.sh data/sre19_av_v_eval_test v_eval ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre19

fi

score_plda_dir=$score_dir/cosine_snorm${ncoh}_v1


if [ $stage -le 9 ]; then

    #SRE19
    echo "SRE19 S-Norm"
    steps_be/eval_face_vid_be_snorm_v1.sh --cmd "$train_cmd" --ncoh $ncoh \
    	data/sre19_av_v_dev_test/trials \
    	data/sre19_av_v_dev_enroll/utt2spk \
    	$face_embed_ref_dir/sre19_av_v_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_dev_test/embed.scp \
	data/${coh_data}/utt2spk \
	$face_embed_facedet_dir/${coh_data}/embed.scp \
    	$score_plda_dir/sre19_av_v_dev_scores &

    
    steps_be/eval_face_vid_be_snorm_v1.sh --cmd "$train_cmd" --ncoh $ncoh \
    	data/sre19_av_v_eval_test/trials \
    	data/sre19_av_v_eval_enroll/utt2spk \
    	$face_embed_ref_dir/sre19_av_v_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/sre19_av_v_eval_test/embed.scp \
	data/${coh_data}/utt2spk \
	$face_embed_facedet_dir/${coh_data}/embed.scp \
    	$score_plda_dir/sre19_av_v_eval_scores &

    wait

    local/score_sre19av.sh data/sre19_av_v_dev_test v_dev $score_plda_dir
    local/score_sre19av.sh data/sre19_av_v_eval_test v_eval $score_plda_dir

fi


if [ $stage -le 10 ]; then

    #JANUS
    echo "JANUS S-Norm"

    steps_be/eval_face_vid_be_snorm_v1.sh --cmd "$train_cmd" --ncoh $ncoh --ncoh-discard 100 \
    	data/janus_dev_test_core/trials \
    	data/janus_dev_enroll/utt2spk \
    	$face_embed_ref_dir/janus_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_dev_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_dev_test_core/embed.scp \
	data/${coh_data}/utt2spk \
	$face_embed_facedet_dir/${coh_data}/embed.scp \
    	$score_plda_dir/janus_dev_core_scores &

    
    steps_be/eval_face_vid_be_snorm_v1.sh --cmd "$train_cmd" --ncoh $ncoh \
    	data/janus_eval_test_core/trials \
    	data/janus_eval_enroll/utt2spk \
    	$face_embed_ref_dir/janus_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_eval_enroll/embed.scp \
    	$face_embed_facedet_dir/janus_eval_test_core/embed.scp \
	data/${coh_data}/utt2spk \
	$face_embed_facedet_dir/${coh_data}/embed.scp \
    	$score_plda_dir/janus_eval_core_scores &

    wait

    local/score_janus_core.sh data/janus_dev_test_core dev $score_plda_dir
    local/score_janus_core.sh data/janus_eval_test_core eval $score_plda_dir

fi


if [ $stage -le 11 ];then
    local/calibrate_sre19av_v_v1_sre19.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sre19av.sh data/sre19_av_v_dev_test v_dev ${score_plda_dir}_cal_v1_sre19
    local/score_sre19av.sh data/sre19_av_v_eval_test v_eval ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_dev_test_core dev ${score_plda_dir}_cal_v1_sre19
    local/score_janus_core.sh data/janus_eval_test_core eval ${score_plda_dir}_cal_v1_sre19

    exit
fi


    
exit

