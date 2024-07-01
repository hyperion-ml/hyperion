# ECAPA-TDNN small

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=adi17

# x-vector cfg
nnet_type=resnet1d
nnet_stages=1
nnet_s1_base_cfg=conf/train_ecapatdnn512x3_xvec_stage1_v3.0.yaml

nnet_name=${feat_type}_ecapatdnn512x3.v3.0
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0040.pth

