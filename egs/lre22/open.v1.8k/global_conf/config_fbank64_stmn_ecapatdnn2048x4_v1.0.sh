# acoustic features
feat_config=conf/fbank64_stmn_8k.yaml
feat_type=fbank64_stmn

#vad
vad_config=conf/vad_8k.yaml

# x-vector training 
nnet_data=open

# x-vector cfg

nnet_type=resnet1d
nnet_stages=1
nnet_s1_base_cfg=conf/train_ecapatdnn2048x4_xvec_stage1_v1.0.yaml

nnet_name=${feat_type}_ecapatdnn2048x4_v1.0
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0012.pth
