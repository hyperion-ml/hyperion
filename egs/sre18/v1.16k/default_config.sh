#Default configuration parameters for the experiment

#xvector training 
nnet_data=train_combined
nnet_vers=1a
nnet_name=1a
nnet_name=ftdnn17m
nnet_num_epochs=3
nnet_dir=exp/xvector_nnet_$nnet_name


#diarization back-end
lda_diar_dim=120
plda_diar_data=voxceleb
be_diar_name=lda${lda_diar_dim}_plda_${plda_diar_data}

diar_thr=-0.9
min_dur=10
rttm_dir=./exp/diarization/$nnet_name/$be_diar_name
diar_name=diar${nnet_name}_thr${diar_thr}


#spk det back-end
lda_tel_dim=150
lda_vid_dim=200
ncoh_tel=400
ncoh_vid=500
ncoh_vast=120

w_mu1=1
w_B1=0.75
w_W1=0.75
w_mu2=1
w_B2=0.6
w_W2=0.6
num_spks=975

plda_tel_y_dim=125
plda_tel_z_dim=150
plda_vid_y_dim=150
plda_vid_z_dim=200

coh_vid_data=sitw_sre18_dev_vast_${diar_name}
coh_vast_data=sitw_sre18_dev_vast_${diar_name}
coh_tel_data=sre18_dev_unlabeled
plda_tel_data=sre_tel_combined
plda_tel_type=splda
plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt_v1_a1_mu${w_mu1}B${w_B1}W${w_W1}_a2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}

plda_vid_data=voxceleb_combined
plda_vid_type=splda
plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v1

be_tel_name=lda${lda_tel_dim}_${plda_tel_label}_${plda_tel_data}
be_vid_name=lda${lda_vid_dim}_${plda_vid_label}_${plda_vid_data}
