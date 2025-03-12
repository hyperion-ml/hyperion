# Res2Net50 trained on sre21 setup

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=""

# x-vector cfg
nnet_type=resnet
nnet_name=${feat_type}_ecapatdnn2048x4_sre21setup.v1

nnet_s1_base_cfg=""
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0030.pth

nnet_s2_base_cfg=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0007.pth

max_nnet_stage=2
xvec_chunk_length=100.0
xvec_diar_chunk_length=90.0

# Back-end
be_sre21_cfg=conf/plda_adapt_sre21_nnet_sre21setup.yaml
be_sre21_name=plda_adapt_sre21_nnet_sre21setup
be_sre24_cfg=conf/plda_adapt_sre24_nnet_sre21setup.yaml
be_sre24_name=plda_adapt_sre24_nnet_sre21setup

# LID using LRE22 Open network
lid_nnet_s1_name=fbank64_stmn_fwseres2net50s8_v1.0.s1
lid_nnet_s1_dir=exp/lid_nnets/$lid_nnet_s1_name
lid_nnet_s1=$lid_nnet_s1_dir/model_ep0012.pth

# Diarization
diar_sre24_cfg=conf/diarization_sre24.yaml
diar_label=diarization
