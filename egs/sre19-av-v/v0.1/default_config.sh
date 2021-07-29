#Default configuration parameters for the experiment


face_det_modeldir=exp/face_models/retina-50
face_reco_modeldir=exp/face_models/r100-arcface
face_det_model=$face_det_modeldir/R50
face_reco_model=$face_reco_modeldir/model

face_embed_name=r100-v4
face_embed_dir=`pwd`/exp/face_embed/$face_embed_name

# #diarization back-end
# lda_diar_dim=120
# plda_diar_data=voxceleb
# be_diar_name=lda${lda_diar_dim}_plda_${plda_diar_data}

# diar_thr=-0.9
# min_dur=10
# rttm_dir=./exp/diarization/$nnet_name/$be_diar_name
# diar_name=diar${nnet_name}_thr${diar_thr}


# #spk det back-end
# lda_vid_dim=200
# ncoh_vid=500
# ncoh_vast=120

# plda_vid_y_dim=150
# plda_vid_z_dim=200

# coh_vid_data=sitw_sre18_dev_vast_${diar_name}
# coh_vast_data=sitw_sre18_dev_vast_${diar_name}

# plda_vid_data=voxceleb_combined
# plda_vid_type=splda
# plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v1

# be_vid_name=lda${lda_vid_dim}_${plda_vid_label}_${plda_vid_data}
