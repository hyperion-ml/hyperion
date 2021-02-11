# LResNet34 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank80_stmn_16k.pyconf
feat_type=fbank80_stmn

# x-vector training 
nnet_data=voxcelebcat
nnet_num_augs=6
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yml --val-aug-cfg conf/reverb_noise_aug.yml"

batch_size_1gpu=32
eff_batch_size=512 # effective batch size
ipe=$nnet_num_augs
min_chunk=4
max_chunk=4
lr=0.05

nnet_type=resnet34 
dropout=0
embed_dim=256

s=30
margin_warmup=20
margin=0.3

nnet_opt="--resnet-type $nnet_type --in-feats 80 --in-channels 1 --in-kernel-size 3 --in-stride 1 --no-maxpool"

opt_opt="--opt-optimizer adam --opt-lr $lr --opt-beta1 0.9 --opt-beta2 0.95 --opt-weight-decay 1e-5 --opt-amsgrad --use-amp"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 10000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"

nnet_name=${feat_type}_${nnet_type}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=70
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth


# back-end
plda_aug_config=conf/reverb_noise_aug.yml
plda_num_augs=6
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxcelebcat
else
    plda_data=voxcelebcat_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200

