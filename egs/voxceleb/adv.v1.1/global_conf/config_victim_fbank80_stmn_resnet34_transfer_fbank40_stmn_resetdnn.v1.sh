# Victim model ResNet34 x-vector
# For the black-box attacks we use Residual E-TDNN to generate the attack and transfer them to the ResNet34
# Both models uses the same features: 80 fbanks
# Both models uses the same training data.

# victim acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# victim x-vector training 
nnet_data=voxceleb2cat_train
nnet_num_augs=6
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

batch_size_1gpu=32
eff_batch_size=512 # effective batch size
min_chunk=4
max_chunk=4
ipe=1
lr=0.05

nnet_type=resnet34
dropout=0
embed_dim=256

s=30
margin_warmup=20
margin=0.3

nnet_opt="--resnet-type $nnet_type --in-feats 80 --in-channels 1 --in-kernel-size 3 --in-stride 1 --no-maxpool"
opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
nnet_name=${feat_type}_${nnet_type}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=70
num_augs=5
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth


# transfer model acoustic features
transfer_feat_config=conf/fbank40_stmn_16k.yaml
transfer_feat_type=fbank40_stmn

# transfer model training
transfer_nnet_data=voxceleb2cat_train #this can be voxceleb2cat or voxceleb1cat
transfer_nnet_num_augs=6
transfer_aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

transfer_batch_size_1gpu=128
transfer_eff_batch_size=512 # effective batch size
transfer_min_chunk=4
transfer_max_chunk=4
transfer_ipe=$transfer_nnet_num_augs
transfer_lr=0.05

transfer_nnet_type=resetdnn
transfer_num_layers=5
transfer_layer_dim=512
transfer_expand_dim=1536
transfer_dilation="1 2 3 4 1"
transfer_kernel_sizes="5 3 3 3 1"
transfer_dropout=0.1
transfer_embed_dim=256

transfer_s=30
transfer_margin_warmup=20
transfer_margin=0.3

transfer_nnet_opt="--tdnn-type $transfer_nnet_type --in-feats 40 --num-enc-blocks $transfer_num_layers --enc-hid-units $transfer_layer_dim --enc-expand-units $transfer_expand_dim --kernel-size $transfer_kernel_sizes --dilation $transfer_dilation"
transfer_opt_opt="--optim.opt-type adam --optim.lr $transfer_lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
transfer_lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
transfer_nnet_name=${transfer_feat_type}_${transfer_nnet_type}_nl${transfer_num_layers}ld${transfer_layer_dim}_e${transfer_embed_dim}_arcs${transfer_s}m${transfer_margin}_do${transfer_dropout}_adam_lr${transfer_lr}_b${transfer_eff_batch_size}_amp.v1
transfer_nnet_num_epochs=70

transfer_nnet_dir=exp/xvector_nnets/$transfer_nnet_name
transfer_nnet=$transfer_nnet_dir/model_ep0070.pth


