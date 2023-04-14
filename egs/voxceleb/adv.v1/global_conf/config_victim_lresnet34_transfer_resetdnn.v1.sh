# Victim model Light ResNet34 x-vector
# For the black-box attacks we use Residual E-TDNN to generate the attack and transfer them to the ResNet34
# Both models uses the same features: 80 fbanks
# Both models uses the same training data.

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# victim x-vector training 
nnet_data=voxceleb2cat_train

# victim x-vector cfg
nnet_type=resnet
nnet_name=${feat_type}_lresnet34
nnet_cfg=conf/train_lresnet34_xvec.yaml
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth

# transfer model training
transfer_nnet_data=voxceleb2cat_train #this can be voxceleb2cat or voxceleb2cat_combined

transfer_nnet_type=resetdnn
transfer_nnet_cfg=train_resetdnn_xvec.yaml
transfer_nnet_name=${feat_type}_resetdnn5x512
transfer_nnet_dir=exp/xvector_nnets/$transfer_nnet_name
transfer_nnet=$transfer_nnet_dir/model_ep0070.pth


