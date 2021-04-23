# LResNet34 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

# x-vector training 
nnet_data=voxcelebcat
nnet_num_augs=6
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

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

opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 10000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"

nnet_name=${feat_type}_${nnet_type}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=70
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth

#xvector finetuning
ft_batch_size_1gpu=16
ft_eff_batch_size=128 # effective batch size
ft_min_chunk=10
ft_max_chunk=30
ft_ipe=1
ft_lr=0.01
ft_nnet_num_epochs=40
ft_margin=0.3
ft_margin_warmup=3
ft_margin=0.5
ft_margin_warmup=5
ft_opt_opt="--optim.opt-type sgd --optim.lr $ft_lr --optim.momentum 0.9 --optim.weight-decay 1e-5 --use-amp --var-batch-size"
ft_lrs_opt="--lrsched.lrsch-type cos_lr --lrsched.t 2500 --lrsched.t-mul 2 --lrsched.warm-restarts --lrsched.gamma 0.75 --lrsched.min-lr 1e-4 --lrsched.warmup-steps 100 --lrsched.update-lr-on-opt-step"
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v1
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0028.pth

# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
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

diar_plda_num_augs=0
if [ $diar_plda_num_augs -eq 0 ]; then
    diar_plda_data=voxcelebcat
else
    diar_plda_data=voxcelebcat_augx${plda_num_augs}
fi
diar_plda_type=splda
diar_lda_dim=150
diar_plda_y_dim=150
diar_plda_z_dim=150

diar_plda_name=lda${diar_lda_dim}_${diar_plda_type}y${diar_plda_y_dim}_v1_${diar_plda_data}
diar_thr=-7
diar_dir=exp/diarization/$nnet_name/${diar_plda_name}/ahc_pcar1_thr${diar_thr}
diar_name=diar_resnet34_thr${diar_thr}
