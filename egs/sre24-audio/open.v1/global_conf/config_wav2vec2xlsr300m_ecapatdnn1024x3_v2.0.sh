# Wav2Vec2 Multilingual 300M params

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=sre96-21_vox

# x-vector cfg

nnet_type=hf_wav2vec2resnet1d

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn1024x3_stage1_v2.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_ecapatdnn1024x3_v2.0
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0055.pth

nnet_s2_base_cfg=conf/train_wavlmlarge_ecapatdnn1024x3_stage2_v2.0.yaml
nnet_name=${hf_model_name}_ecapatdnn1024x3_v2.0
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0031.pth
nnet_s2=$nnet_s2_dir/model_ep0027.pth

nnet_s3_base_cfg=conf/train_wavlmlarge_ecapatdnn1024x3_stage3_v2.0.yaml
nnet_name=${hf_model_name}_ecapatdnn1024x3_v2.0
nnet_s3_name=${nnet_name}.s3
nnet_s3_dir=exp/xvector_nnets/$nnet_s3_name
nnet_s3=$nnet_s3_dir/model_ep0003.pth
#nnet_s3=$nnet_s3_dir/model_ep0007.pth

# nnet_s4_base_cfg=conf/train_wavlmlarge_ecapatdnn1024x3_stage4_v2.0.yaml
# nnet_s4_name=${nnet_name}.s4
# nnet_s4_dir=exp/xvector_nnets/$nnet_s4_name
# nnet_s4=$nnet_s4_dir/model_ep0003.pth

max_nnet_stage=3
xvec_chunk_length=100.0
xvec_diar_chunk_length=90.0

# Back-end
be_sre21_cfg=conf/plda_adapt_sre21.yaml
be_sre21_name=plda_adapt_sre21
be_sre24_cfg=conf/plda_adapt_sre24.yaml
be_sre24_name=plda_adapt_sre24

# LID using LRE22 Open network
lid_nnet_s1_name=fbank64_stmn_fwseres2net50s8_v1.0.s1
lid_nnet_s1_dir=exp/lid_nnets/$lid_nnet_s1_name
lid_nnet_s1=$lid_nnet_s1_dir/model_ep0012.pth

# Diarization
diar_sre24_cfg=conf/diarization_sre24.yaml
diar_label=diarization
