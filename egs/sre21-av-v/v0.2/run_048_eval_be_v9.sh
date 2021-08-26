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
self_att_a=2

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 


be_vid_name=be_v9_selfatt_a${self_att_a}

face_embed_ref_dir=$face_embed_dir/ref
face_embed_facedet_dir=$face_embed_dir/facedet

be_vid_dir=exp/be/$face_embed_name/$be_vid_name
score_dir=exp/scores/$face_embed_name/${be_vid_name}
score_plda_dir=$score_dir/cosine

if [ $stage -le 4 ]; then

  echo "SRE19"
  steps_be/eval_face_vid_be_v9.sh \
    --cmd "$train_cmd" \
    --self-att-a $self_att_a \
    data/sre19_av_v_dev_test/trials \
    data/sre19_av_v_dev_enroll/utt2spk \
    $face_embed_ref_dir/sre19_av_v_dev_enroll/embed.scp \
    $face_embed_facedet_dir/sre19_av_v_dev_enroll/embed.scp \
    $face_embed_facedet_dir/sre19_av_v_dev_test/embed.scp \
    $score_plda_dir/sre19_av_v_dev_scores &
  
  steps_be/eval_face_vid_be_v9.sh \
    --cmd "$train_cmd" \
    --self-att-a $self_att_a \
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
  steps_be/eval_face_vid_be_v9.sh \
    --cmd "$train_cmd" \
    --self-att-a $self_att_a \
    data/janus_dev_test_core/trials \
    data/janus_dev_enroll/utt2spk \
    $face_embed_ref_dir/janus_dev_enroll/embed.scp \
    $face_embed_facedet_dir/janus_dev_enroll/embed.scp \
    $face_embed_facedet_dir/janus_dev_test_core/embed.scp \
    $score_plda_dir/janus_dev_core_scores &
  
  steps_be/eval_face_vid_be_v9.sh --cmd "$train_cmd" \
				  --self-att-a $self_att_a \
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
  steps_be/eval_face_vid_be_snorm_v9.sh --cmd "$train_cmd" --ncoh $ncoh \
					--self-att-a $self_att_a \
    					data/sre19_av_v_dev_test/trials \
    					data/sre19_av_v_dev_enroll/utt2spk \
    					$face_embed_ref_dir/sre19_av_v_dev_enroll/embed.scp \
    					$face_embed_facedet_dir/sre19_av_v_dev_enroll/embed.scp \
    					$face_embed_facedet_dir/sre19_av_v_dev_test/embed.scp \
					data/${coh_data}/utt2spk \
					$face_embed_facedet_dir/${coh_data}/embed.scp \
    					$score_plda_dir/sre19_av_v_dev_scores &
  
  steps_be/eval_face_vid_be_snorm_v9.sh \
    --cmd "$train_cmd" --ncoh $ncoh \
    --self-att-a $self_att_a \
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
  steps_be/eval_face_vid_be_snorm_v9.sh \
    --cmd "$train_cmd" --ncoh $ncoh --ncoh-discard 100 \
    --self-att-a $self_att_a \
    data/janus_dev_test_core/trials \
    data/janus_dev_enroll/utt2spk \
    $face_embed_ref_dir/janus_dev_enroll/embed.scp \
    $face_embed_facedet_dir/janus_dev_enroll/embed.scp \
    $face_embed_facedet_dir/janus_dev_test_core/embed.scp \
    data/${coh_data}/utt2spk \
    $face_embed_facedet_dir/${coh_data}/embed.scp \
    $score_plda_dir/janus_dev_core_scores &

  steps_be/eval_face_vid_be_snorm_v9.sh \
    --cmd "$train_cmd" --ncoh $ncoh \
    --self-att-a $self_att_a \
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
fi



exit

