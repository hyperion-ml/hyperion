# VQ-VAE with Transformer Encoder for Enc and Dec with 
# 6 transformer blocks, d_model=512, heads=8, d_ff=2048, latent_dim=512, codebook=8x8, compression factor=36

nnet_data=voxceleb2cat_train
batch_size_1gpu=16
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.01

model_type=vq-vae

dropout=0
narch=transformer-enc-v1
blocks=6
d_model=512
heads=8
d_ff=2048

latent_dim=512
vq_type=multi-ema-k-means-vq
vq_clusters=512
num_groups=8

vae_opt="--in-feats 80 --z-dim $latent_dim --vq-type $vq_type --vq-clusters $vq_clusters --vq-groups $num_groups"
enc_opt="--enc.num-blocks $blocks --enc.d-model $d_model --enc.num-heads $heads --enc.ff-type linear --enc.d-ff $d_ff --enc.in-layer-type linear --enc.att-type scaled-dot-prod-v1"
dec_opt="--dec.in-feats $latent_dim --dec.num-blocks $blocks --dec.d-model $d_model --dec.num-heads $heads --dec.ff-type linear --dec.d-ff $d_ff --dec.in-layer-type linear --dec.att-type scaled-dot-prod-v1"


#opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
#lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 12000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 2000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
nnet_name=${model_type}_${narch}_b${blocks}d${d_model}h${heads}linff${d_ff}_${vq_type}_z${latent_dim}c${vq_clusters}x${num_groups}_do${dropout}_optv4_adam_lr${lr}_b${eff_batch_size}.$nnet_data
nnet_num_epochs=160
num_augs=5
nnet_dir=exp/vae_nnets/$nnet_name
nnet=$nnet_dir/model_ep0160.pth

# xvector network trained with recipe v1.1
xvec_nnet_name=fbank80_stmn_lresnet34_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1
xvec_nnet_dir=../v1.1/exp/xvector_nnets/$xvec_nnet_name
xvec_nnet=$xvec_nnet_dir/model_ep0070.pth


