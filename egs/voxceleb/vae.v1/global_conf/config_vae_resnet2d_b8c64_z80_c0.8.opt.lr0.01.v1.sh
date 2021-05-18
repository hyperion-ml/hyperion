# VAE with symmetric ResNet2D encoder-decoder with 
# 8 residual blocks, 64 base channels , latent_channels=80, compression factor=0.8

nnet_data=voxceleb2cat_train
batch_size_1gpu=16
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.01
dropout=0
latent_dim=80
model_type=vae
narch=resnet2d
vae_opt=""
enc_opt="--enc.in-conv-channels 64 --enc.in-kernel-size 5 --enc.in-stride 1 --enc.resb-repeats 2 2 2 2 --enc.resb-channels 64 128 256 512 --enc.resb-kernel-sizes 3 --enc.resb-strides 1 2 2 2"
dec_opt="--dec.in-channels 80 --dec.in-conv-channels 512 --dec.in-kernel-size 3 --dec.in-stride 1 --dec.resb-repeats 2 2 2 2 --dec.resb-channels 512 256 128 64 --dec.resb-kernel-sizes 3 --dec.resb-strides 1 2 2 2"

opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 16000 --lrsched.hold-steps 16000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 8000 --lrsched.update-lr-on-opt-step"
nnet_name=${model_type}_${narch}_b8c64_z${latent_dim}_c0.8_do${dropout}_optv1_adam_lr${lr}_b${eff_batch_size}.$nnet_data
nnet_num_epochs=205
num_augs=5
nnet_dir=exp/vae_nnets/$nnet_name
nnet=$nnet_dir/model_ep0205.pth

# xvector network trained with recipe v1.1
xvec_nnet_name=fbank80_stmn_lresnet34_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1
xvec_nnet_dir=../v1.1/exp/xvector_nnets/$xvec_nnet_name
xvec_nnet=$xvec_nnet_dir/model_ep0070.pth

