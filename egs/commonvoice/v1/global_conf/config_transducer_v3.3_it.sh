# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
# vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=it_train_proc_audio
dev_data=it_dev_proc_audio
test_data=it_test_proc_audio

language=it

bpe_model=data/it_lang_bpe_1000/bpe.model
# x-vector cfg

nnet_type=hf_wav2vec2transducer

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage1_v3.3.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_transducer_v3.3_it
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/transducer_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0015.pth

nnet_s2_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage2_v3.3.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/transducer_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0020.pth

nnet_s3_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage1_v1.0.yaml
nnet_s3_args=""
nnet_s3_name=${nnet_name}.s3
nnet_s3_dir=exp/transducer_nnets/$nnet_s3_name
nnet_s3=$nnet_s3_dir/model_ep0002.pth
nnet_s3=$nnet_s3_dir/model_ep0005.pth
