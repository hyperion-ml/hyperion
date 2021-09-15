# Time SE Res2Net50 w26s4 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train
nnet_num_augs=6
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

batch_size_1gpu=32
eff_batch_size=512 # effective batch size
ipe=$nnet_num_augs
min_chunk=4
max_chunk=4
lr=0.05

nnet_type=resnet1d
block_type=seres2bn # squeeze-excitation res2net bottleneck
channels=512
ep_channels=1536
width_factor=1
scale=8
se_r=4
dropout=0

attstats_inner=128
embed_dim=256
s=30
margin_warmup=20
margin=0.3

nnet_opt="--resnet_enc.in-feats 80 \
		     --resnet_enc.in-conv-channels $channels \
		     --resnet_enc.in-kernel-size 5 \
		     --resnet_enc.in-stride 1 \
		     --resnet_enc.resb-type $block_type \
		     --resnet_enc.resb-repeats 1 1 1 \
		     --resnet_enc.resb-channels $channels \
		     --resnet_enc.resb-kernel-sizes 3 \
		     --resnet_enc.resb-dilations 2 3 4 \
		     --resnet_enc.resb-strides 1 \
		     --resnet_enc.res2net-width-factor $width_factor \
		     --resnet_enc.res2net-scale $scale \
		     --resnet_enc.se-r $se_r \
		     --resnet_enc.multilayer \
                     --resnet_enc.multilayer-concat \
                     --resnet_enc.endpoint-channels $ep_channels \
		     --pool_net.pool-type ch-wise-att-mean+stddev \
		     --pool_net.inner-feats $attstats_inner \
		     --embed-dim $embed_dim"

opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"

nnet_name=${feat_type}_ecapatdnn512x3_chattstatsi128_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=70
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

