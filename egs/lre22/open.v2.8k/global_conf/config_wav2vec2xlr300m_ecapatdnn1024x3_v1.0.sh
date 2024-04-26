# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 1024x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
vad_config=conf/vad_8k.yaml

# x-vector training 
nnet_data=open

# x-vector cfg
nnet_stages=2
nnet_type=hf_wav2vec2resnet1d

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn1024x3_stage1_v1.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_ecapatdnn1024x3_v1.0
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0012.pth

nnet_s2_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn1024x3_stage2_v1.0.yaml
nnet_s2_args=""
nnet_name=${hf_model_name}_ecapatdnn1024x3_v1.0
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0006.pth

