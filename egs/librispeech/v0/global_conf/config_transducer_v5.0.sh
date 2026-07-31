# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2base
#vad
# vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=train_clean_100
dev_data=dev_clean

bpe_model=data/lang_bpe_1000/bpe.model
# x-vector cfg

nnet_type=hf_wav2vec2transducer

nnet_s1_base_cfg=conf/train_wav2vec2base_transducer_stage1_v5.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_transducer_v5.0
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/transducer_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0030.pth
nnet_s1=$nnet_s1_dir/model_ep0050.pth
nnet_s1=$nnet_s1_dir/model_ep0075.pth
nnet_s1=$nnet_s1_dir/model_ep0106.pth

nnet_s2_base_cfg=conf/train_wav2vec2base_transducer_stage2_v1.0.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/transducer_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0020.pth
