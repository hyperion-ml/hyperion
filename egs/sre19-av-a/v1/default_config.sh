#Default configuration parameters for the experiment

#xvector training 
nnet_data=voxceleb_combined
nnet_vers=3a.1
nnet_name=3a.1.vcc
#nnet_name=ftdnn17m
nnet_num_epochs=6
num_augs=5
nnet_dir=exp/xvector_nnet_$nnet_name

#diarization back-end
lda_diar_dim=120
plda_diar_data=voxceleb
be_diar_name=lda${lda_diar_dim}_plda_${plda_diar_data}

diar_thr=-0.9
min_dur_spkdet_subsegs=10 # minimum duration for the diarization clusters used for spk detection
rttm_dir=./exp/diarization/$nnet_name/$be_diar_name
diar_name=diar${nnet_name}_thr${diar_thr}


#spk det back-end
lda_vid_dim=200
ncoh_vid=500
ncoh_vast=120

plda_vid_y_dim=150
plda_vid_z_dim=200

coh_vid_data=sitw_sre18_dev_vast_${diar_name}
coh_vast_data=sitw_sre18_dev_vast_${diar_name}

plda_vid_data=voxceleb_combined
plda_vid_type=splda
plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v1

be_vid_name=lda${lda_vid_dim}_${plda_vid_label}_${plda_vid_data}
