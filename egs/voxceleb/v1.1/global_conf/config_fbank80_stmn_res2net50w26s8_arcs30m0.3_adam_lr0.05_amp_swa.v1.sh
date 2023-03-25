# Res2Net50 w26s8 x-vector with mixed precision training and SWA

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg

nnet_type=resnet

resnet_type=res2net50
batch_size_1gpu=16
eff_batch_size=512 # effective batch size
dropout=0
embed_dim=256
lr=0.05
s=30
margin_warmup=20
margin=0.3
width_factor=3.25
scale=8
ws_tag=w26s8
nnet_num_epochs=90

nnet_s1_base_cfg=conf/train_res2net50_xvec_default.yaml
nnet_s1_args="--data.train.sampler.batch-size $batch_size_1gpu --model.resnet-type $resnet_type --model.res2net-width-factor $width_factor --model.res2net-scale $scale --trainer.epochs $nnet_num_epochs --trainer.swa-start 70 --trainer.swa-lr 1e-3 --trainer.swa-anneal-epochs 5"

nnet_s1_name=${feat_type}_${resnet_type}${ws_tag}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp_swa.v1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/swa_model_ep0091.pth

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

