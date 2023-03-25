# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
# vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=en_fr_it_train_proc_audio
dev_data=en_fr_it_dev_proc_audio
test_data="en_test_proc_audio fr_test_proc_audio it_test_proc_audio"

lans="en fr it"
language=en_fr_it

bpe_model=data/en_fr_it_lang_bpe_2000/bpe.model
# x-vector cfg

nnet_type=hf_wav2vec2resnet1d

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn512x3_stage1_v1.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_resnet1d_v3.3_en_fr_it
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/resnet1d_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0022.pth

nnet_s2_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn512x3_stage2_v1.0.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/resnet1d_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0020.pth

nnet_s3_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn512x3_stage3_v1.0.yaml
nnet_s3_args=""
nnet_s3_name=${nnet_name}.s3
nnet_s3_dir=exp/resnet1d_nnets/$nnet_s3_name
nnet_s3=$nnet_s3_dir/model_ep0002.pth
nnet_s3=$nnet_s3_dir/model_ep0005.pth