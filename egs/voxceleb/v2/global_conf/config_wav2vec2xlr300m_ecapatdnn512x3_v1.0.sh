# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg

nnet_type=hf_wav2vec2resnet1d

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn512x3_stage1_v1.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_ecapatdnn512x3_v1.0
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0060.pth

nnet_s2_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn512x3_stage2_v1.0.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0020.pth

nnet_s3_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn512x3_stage3_v1.0.yaml
nnet_s3_args=""
nnet_s3_name=${nnet_name}.s3
nnet_s3_dir=exp/xvector_nnets/$nnet_s3_name
nnet_s3=$nnet_s3_dir/model_ep0002.pth

# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxceleb2cat_train
else
    plda_data=voxceleb2cat_train_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200

