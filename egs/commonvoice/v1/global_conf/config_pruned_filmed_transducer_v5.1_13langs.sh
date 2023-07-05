# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
# vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=13_langs_train_proc_audio
dev_data=13_langs_dev_proc_audio
test_data="ga-IE_test_proc_audio br_test_proc_audio sl_test_proc_audio cv_test_proc_audio tt_test_proc_audio tr_test_proc_audio cy_test_proc_audio it_test_proc_audio kab_test_proc_audio fr_test_proc_audio de_test_proc_audio ca_test_proc_audio en_test_proc_audio" 

lans="sl ga-IE cv br tr cy tt ca kab de fr it en"
language=13_langs_weighted

# bpe_model=data/13_langs_lang_bpe_4000/bpe.model
bpe_model=data/13_langs_weighted_lang_bpe_8000/bpe.model
# bpe_model=data/13_langs_weighted_lang_bpe_16000/bpe.model



nnet_type=hf_wav2vec2rnn_filmed_transducer

nnet_s1_base_cfg=conf/train_wav2vec2base_rnnt_film_k2_pruned_stage3_v5.1.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_rnnt_k2_pruned_film.v5.1_13_langs_weighted_8000_bpe
nnet_s1_name=$nnet_name.s3

nnet_s1_dir=exp/transducer_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0042.pth

nnet_s2_base_cfg=conf/train_wav2vec2base_rnnt_film_k2_pruned_stage4_v5.1.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s4
nnet_s2_dir=exp/transducer_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0003.pth

nnet_s3_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage1_v5.1.yaml
nnet_s3_args=""
nnet_s3_name=${nnet_name}.s5
nnet_s3_dir=exp/transducer_nnets/$nnet_s3_name
nnet_s3=$nnet_s3_dir/model_ep0002.pth
nnet_s3=$nnet_s3_dir/model_ep0005.pth
