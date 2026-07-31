# FwSERes2Net

# acoustic features
feat_config=conf/fbank64_stmn_nb_16k.yaml
feat_type=fbank64_stmn_nb

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=sre96-12_16_21

# x-vector cfg
nnet_type=resnet
nnet_name=${feat_type}_fwseres2net50w26s8.v3.2

nnet_s1_base_cfg=conf/train_${feat_type}_fwseres2net50w26s8_xvec_stage1_v3.2.yaml
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0127.pth

nnet_s2_base_cfg=conf/train_${feat_type}_fwseres2net50w26s8_xvec_stage2_v3.2.yaml
nnet_name=${feat_type}_fwseres2net50w26s8.v3.2
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0003.pth

max_nnet_stage=2
xvec_chunk_length=100.0
xvec_diar_chunk_length=90.0

# Back-end
be_sre21_cfg=conf/plda_adapt_sre21_nnet_sre21setup.yaml
be_sre21_name=plda_adapt_sre21_nnet_sre21setup
be_sre24_cfg=conf/plda_adapt_sre24.yaml
be_sre24_name=plda_adapt_sre24

# LID training 
lid_nnet_type=resnet
lid_feat_type=fbank64_stmn_nb
lid_nnet_name=${lid_feat_type}_fwseresnet34.v1.0

lid_nnet_s1_cfg=conf/train_${lid_feat_type}_fwseresnet34_lid_stage1_v1.0.yaml
lid_nnet_s1_name=$lid_nnet_name.s1
lid_nnet_s1_dir=exp/lid_nnets/$lid_nnet_s1_name
lid_nnet_s1=$lid_nnet_s1_dir/model_ep0035.pth

# Diarization
diar_sre24_cfg=conf/diarization_sre24.yaml
diar_label=diarization
