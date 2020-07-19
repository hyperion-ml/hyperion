# VAE with symmetric ResNet1D encoder-decoder with 
# 8 residual blocks, 256 dim per block, latent_dim=80, compression factor=8

nnet_data=voxceleb2cat
batch_size_1gpu=128
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.01
dropout=0
latent_dim=80
model_type=vae
narch=resnet1d
vae_opt="--in-feats 80"
enc_opt="--enc-in-conv-channels 256 --enc-in-kernel-size 5 --enc-in-stride 1 --enc-resb-repeats 1 2 3 2 --enc-resb-channels 256 --enc-resb-kernel-sizes 3 --enc-resb-strides 1 2 2 2"
dec_opt="--dec-in-channels 80 --dec-in-conv-channels 256 --dec-in-kernel-size 3 --dec-in-stride 1 --dec-resb-repeats 1 2 3 2 --dec-resb-channels 256 --dec-resb-kernel-sizes 3 --dec-resb-strides 1 2 2 2"

opt_opt="--opt-optimizer adam --opt-lr $lr --opt-beta1 0.9 --opt-beta2 0.95 --opt-weight-decay 1e-5 --opt-amsgrad"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"
nnet_name=${model_type}_${narch}_b8d256_z${latent_dim}_c8_do${dropout}_optv1_adam_lr${lr}_b${eff_batch_size}.$nnet_data
nnet_num_epochs=600
num_augs=5
nnet_dir=exp/vae_nnets/$nnet_name
nnet=$nnet_dir/model_ep0422.pth
