# ECAPA-TDNN 2048x4 layers x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxcelebcat_sre_alllangs_mixfs
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

batch_size_1gpu=16
eff_batch_size=512 # effective batch size
ipe=1
min_chunk=4
max_chunk=4
lr=0.02

nnet_type=resnet1d
block_type=seres2bn # squeeze-excitation res2net bottleneck
channels=2048
ep_channels=8192
width_factor=1
scale=8
se_r=16
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
		     --resnet_enc.resb-repeats 1 1 1 1 \
		     --resnet_enc.resb-channels $channels \
		     --resnet_enc.resb-kernel-sizes 3 \
		     --resnet_enc.resb-dilations 2 3 4 5 \
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

opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp --swa-start 65 --swa-lr 1e-3 --swa-anneal-epochs 5"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 10000 --lrsched.hold-steps 35000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"

nnet_name=${feat_type}_ecapatdnn2048x4_chattstatsi128_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=75
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth
nnet=$nnet_dir/swa_model_ep0076.pth

# xvector full net finetuning with out-of-domain
ft_batch_size_1gpu=8
ft_eff_batch_size=128 # effective batch size
ft_min_chunk=10
ft_max_chunk=15
ft_ipe=1
ft_lr=0.01
ft_nnet_num_epochs=15
ft_margin=0.5
ft_margin_warmup=3

ft_opt_opt="--optim.opt-type sgd --optim.lr $ft_lr --optim.momentum 0.9 --optim.weight-decay 1e-5 --use-amp --var-batch-size"
ft_lrs_opt="--lrsched.lrsch-type cos_lr --lrsched.t 2500 --lrsched.t-mul 2 --lrsched.warm-restarts --lrsched.gamma 0.75 --lrsched.min-lr 1e-4 --lrsched.warmup-steps 100 --lrsched.update-lr-on-opt-step"
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_arcm${ft_margin}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v1
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0007.pth

# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxceleb2cat_train
else
    plda_data=voxceleb2cat_train_augx${plda_num_augs}
fi
plda_type=splda
# lda_dim=200
# plda_y_dim=150
# plda_z_dim=200

