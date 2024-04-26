# Conformer + RNN-T

# training data 
nnet_train_data=train_960h
nnet_val__data=dev_all

# tokenizer
bpe_model=data/lang_bpe_1000/bpe.model

# rnn-t cfg
nnet_type=conformer_v1_rnn_transducer
nnet_name=fbank80_mn_conf16x144_rnnt_k2_pruned.v1.0p
nnet_s1_base_cfg=conf/train_${nnet_name}.s1.yaml
nnet_s1_args=""
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=exp/asr_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0115.pth
