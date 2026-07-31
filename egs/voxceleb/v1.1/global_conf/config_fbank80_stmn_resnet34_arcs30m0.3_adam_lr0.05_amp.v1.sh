# ResNet34 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg
nnet_type=resnet

resnet_type=resnet34
batch_size_1gpu=32
eff_batch_size=512 # effective batch size
dropout=0
embed_dim=256
lr=0.05
s=30
margin_warmup=20
margin=0.3
nnet_num_epochs=70

nnet_s1_base_cfg=conf/train_resnet34_xvec_default.yaml
nnet_s1_args="--data.train.sampler.batch-size $batch_size_1gpu"

nnet_s1_name=${feat_type}_${resnet_type}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0070.pth


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

