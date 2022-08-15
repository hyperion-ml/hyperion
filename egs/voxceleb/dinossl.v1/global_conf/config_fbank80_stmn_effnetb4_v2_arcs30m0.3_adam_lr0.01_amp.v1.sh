# EfficientNet-b4 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg
nnet_type=efficientnet

effnet_type=efficientnet-b4
dropout=0
embed_dim=256
s=30
margin_warmup=20
margin=0.3
se_r=4

batch_size_1gpu=16
eff_batch_size=512 # effective batch size
lr=0.01
nnet_num_epochs=70

xvec_train_base_cfg=conf/train_effnetb4_xvec_default.yaml
xvec_train_args="--data.train.sampler.batch-size $batch_size_1gpu --trainer.optim.lr $lr"

nnet_name=${feat_type}_${effnet_type}_is1_mbs1122121_ser${se_r}_fixsh_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1

nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth


# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=6
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxceleb2cat_train
else
    plda_data=voxceleb2cat_train_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200
