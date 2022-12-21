# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
# vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=train_clean_100
dev_data=dev_clean
test_data=test_clean
# nnet_data=train_clean_small

bpe_model=data/lang_bpe_1000/bpe.model
# x-vector cfg

nnet_type=hf_wav2vec2transducer

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage1_v1.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_transducer_v1.0
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/transducer_nnets/$nnet_s1_name
nnet_s1=${nnet_s1_dir}_pre/model_ep0060.pth

nnet_s2_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage1_v2.0.yaml
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
