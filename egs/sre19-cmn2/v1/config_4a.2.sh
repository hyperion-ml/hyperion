# F-TDNN x-vector with 14 layers with 600 dim

#xvector training 
nnet_data=train_combined
nnet_vers=4a.1
nnet_name=4a.2.tc
nnet_num_epochs=6
num_augs=4
nnet_dir=exp/xvector_nnet_$nnet_name

# spk det back-end
plda_tel_data=sre_tel
