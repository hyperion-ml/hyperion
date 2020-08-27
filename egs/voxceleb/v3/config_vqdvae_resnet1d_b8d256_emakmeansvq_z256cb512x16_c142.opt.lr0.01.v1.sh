# VQ-VAE with symmetric ResNet1D encoder-decoder with 
# 8 residual blocks, 256 dim per block, latent_dim=256, codebook=512, compression factor=142

nnet_data=voxceleb2cat_combined
batch_size_1gpu=256
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.01
dropout=0
latent_dim=256
vq_clusters=512
num_groups=16
narch=resnet1d
model_type=vq-dvae
vq_type=multi-ema-k-means-vq
vae_opt="--in-feats 80 --z-dim $latent_dim --vq-type $vq_type --vq-clusters $vq_clusters --vq-groups $num_groups"
enc_opt="--enc-in-conv-channels 256 --enc-in-kernel-size 5 --enc-in-stride 1 --enc-resb-repeats 1 2 3 2 --enc-resb-channels 256 --enc-resb-kernel-sizes 3 --enc-resb-strides 1 2 2 2"
dec_opt="--dec-in-channels 256 --dec-in-conv-channels 256 --dec-in-kernel-size 3 --dec-in-stride 1 --dec-resb-repeats 1 2 3 2 --dec-resb-channels 256 --dec-resb-kernel-sizes 3 --dec-resb-strides 1 2 2 2"

opt_opt="--opt-optimizer adam --opt-lr $lr --opt-beta1 0.9 --opt-beta2 0.95 --opt-weight-decay 1e-5 --opt-amsgrad"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 16000 --lrsch-hold-steps 16000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 8000 --lrsch-update-lr-on-opt-step"
nnet_name=${model_type}_${narch}_b8d256_${vq_type}_z${latent_dim}c${vq_clusters}x${num_groups}_do${dropout}_optv1_adam_lr${lr}_b${eff_batch_size}.$nnet_data
nnet_num_epochs=100
num_augs=5
nnet_dir=exp/vae_nnets/$nnet_name
nnet=$nnet_dir/model_ep0100.pth
