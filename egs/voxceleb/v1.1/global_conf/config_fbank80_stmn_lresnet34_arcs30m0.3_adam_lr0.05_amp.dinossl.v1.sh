# LResNet34 x-vector with mixed precision training
# Edited from global_conf/config_fbank80_stmn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh for dinossl training

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml


# x-vector training 
nnet_data=voxceleb2cat_train
nnet_num_augs=6
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

#batch_size_1gpu=128
batch_size_1gpu=64
#eff_batch_size=512 # effective batch size
eff_batch_size=256
ipe=$nnet_num_augs
min_chunk=2
max_chunk=2
lr=0.05

nnet_type=lresnet34 #light resnet
dropout=0
embed_dim=256

s=30
margin_warmup=20
margin=0.3

nnet_opt="--resnet-type $nnet_type --in-feats 80 --in-channels 1 --in-kernel-size 3 --in-stride 1 --no-maxpool"

opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp --optim.dino-style True"
#lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
lrs_opt="--lrsched.lrsch-type dinossl --lrsched.dinossl_lr 0.005 --lrsched.dinossl_min_lr 1e-6 --lrsched.dinossl_warmup_epochs 10 --lrsched.dinossl_weight_decay 1e-4 --lrsched.dinossl_weight_decay_end 1e-4 --lrsched.dinossl_momentum_teacher 0.996"

nnet_name=${feat_type}_${nnet_type}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=70
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth

# dinossl related
dinossl=true
## dino-head
dinossl_out_dim=65536
dinossl_use_bn_in_head=false
dinossl_norm_last_layer=true
## data-augmentation
dinossl_local_crops_number=4
## teacher temperature
dinossl_warmup_teacher_temp=0.04
dinossl_teacher_temp=0.04
dinossl_warmup_teacher_temp_epochs=0
## chunk sampling related
dinossl_chunk_len_mult=2 # a factor that long chunks increase from short chunks

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

