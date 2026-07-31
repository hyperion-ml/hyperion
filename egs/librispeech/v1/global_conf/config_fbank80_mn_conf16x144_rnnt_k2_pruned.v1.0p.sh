# Conformer + RNN-T

# training data 
nnet_train_data=librispeech_train-960
nnet_val_data=librispeech_dev

# tokenizer
token_train_data=librispeech_train-960
token_cfg=conf/sp_unigram_512.yaml
token_dir=data/token_${token_train_data}_unigram_512
token_model=$token_dir/tokenizer.model

# rnn-t cfg
nnet_type=conformer_v1_rnn_transducer
nnet_name=fbank80_mn_conf16x144_rnnt_k2_pruned.v1.0p
nnet_s1_cfg=conf/train_${nnet_name}.s1.yaml
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/asr_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0115.pth
