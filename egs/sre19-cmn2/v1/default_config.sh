#Default configuration parameters for the experiment
# F-TDNN x-vector with 10 layers with 1024 dim

#xvector training 
nnet_data=train_combined
nnet_vers=3a.1
nnet_name=3a.1.tc
#nnet_name=ftdnn17m
nnet_num_epochs=2
num_augs=4
nnet_dir=exp/xvector_nnet_$nnet_name


# spk det back-end
plda_tel_data=sre_tel
