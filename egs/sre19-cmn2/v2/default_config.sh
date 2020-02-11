#Default configuration parameters for the experiment

#xvector training 
nnet_data=train_combined
nnet_vers=3a.1
nnet_name=3a.1.tc
#nnet_name=ftdnn17m
nnet_num_epochs=5
num_augs=4
nnet_dir=exp/xvector_nnet_$nnet_name


# spk det back-end

# lda_tel_dim=150
# lda_vid_dim=200
# ncoh_tel=400

# w_mu1=1
# w_B1=0.75
# w_W1=0.75
# w_mu2=1
# w_B2=0.6
# w_W2=0.6
# num_spks=975

# plda_tel_y_dim=125
# plda_tel_z_dim=150

#plda_tel_data=sre_tel_combined
plda_tel_data=sre_tel
# plda_tel_type=splda
# plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt_v1_a1_mu${w_mu1}B${w_B1}W${w_W1}_a2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}

# be_tel_name=lda${lda_tel_dim}_${plda_tel_label}_${plda_tel_data}

# coh_tel_data=sre18_dev_unlabeled
