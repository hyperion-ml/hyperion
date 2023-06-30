# Res2Net50 w26s4 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxcelebcat

eff_batch_size=512 # effective batch size
min_chunk=4
max_chunk=4
lr=0.05

nnet_type=resnet
dropout=0
embed_dim=256

s=30
margin_warmup=20
margin=0.3

nnet_base_cfg=conf/train_res2net50w26s4_xvec_stage1_v1.0.yaml
nnet_name=${feat_type}_res2net50w26s4_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1.$nnet_data
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0061.pth

# xvector full net finetuning with out-of-domain
ft_eff_batch_size=128 # effective batch size
ft_min_chunk=10
ft_max_chunk=15
ft_lr=0.01
ft_margin=0.5

ft_nnet_base_cfg=conf/train_res2net50w26s4_xvec_stage2_v1.0.yaml
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_arcm${ft_margin}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v1
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0021.pth

# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
plda_type=splda

