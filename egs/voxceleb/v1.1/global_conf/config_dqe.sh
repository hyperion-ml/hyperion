# ECAPA-TDNN small

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg

nnet_type=resnet1d

batch_size_1gpu=256
eff_batch_size=512 # effective batch size
dropout=0
embed_dim=64
lr=0.05
s=30
margin_warmup=20
margin=0.3
nnet_num_epochs=10

xvec_train_base_cfg=conf/train_ecapatdnn32x2_xvec.yaml
xvec_train_args="--data.train.sampler.batch-size $batch_size_1gpu --trainer.epochs $nnet_num_epochs"

nnet_name=${feat_type}_ecapatdnn32x2_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1

nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0010.pth


# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxceleb2cat_train
else
    plda_data=voxceleb2cat_train_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200

