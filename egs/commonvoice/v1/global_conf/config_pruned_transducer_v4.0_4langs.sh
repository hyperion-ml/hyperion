# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
# vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=4_langs_train_proc_audio
dev_data=4_langs_dev_proc_audio

test_data="tr_test_proc_audio fr_test_proc_audio de_test_proc_audio it_test_proc_audio" 


lans="tr de fr it"
language=13_langs_weighted

# bpe_model=data/13_langs_lang_bpe_4000/bpe.model
bpe_model=data/13_langs_weighted_lang_bpe_8000/bpe.model
# bpe_model=data/13_langs_weighted_lang_bpe_16000/bpe.model
# x-vector cfg

nnet_type=hf_wav2vec2rnn_transducer

nnet_s1_base_cfg=conf/train_wav2vec2base_rnnt_k2_pruned_4langs_stage1_v4.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_rnnt_k2_pruned.v4.0_4_langs_weighted_8000_bpe
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/transducer_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0015.pth

nnet_s2_base_cfg=conf/train_wav2vec2base_rnnt_k2_pruned_stage2_v4.0.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/transducer_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0015.pth

nnet_s3_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage1_v1.0.yaml
nnet_s3_args=""
nnet_s3_name=${nnet_name}.s3
nnet_s3_dir=exp/transducer_nnets/$nnet_s3_name
nnet_s3=$nnet_s3_dir/model_ep0002.pth
nnet_s3=$nnet_s3_dir/model_ep0005.pth