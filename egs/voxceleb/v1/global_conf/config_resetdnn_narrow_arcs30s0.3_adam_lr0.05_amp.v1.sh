# Residual Extended TDNN x-Vector with 5 ETDNN blocks 512 channels/block

#xvector training 
nnet_data=voxceleb2cat_train_combined

batch_size_1gpu=128
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.05

nnet_type=resetdnn
num_layers=5
layer_dim=512
expand_dim=1536
dilation="1 2 3 4 1"
kernel_sizes="5 3 3 3 1"
dropout=0.1
embed_dim=256

s=30
margin_warmup=20
margin=0.3

nnet_opt="--tdnn-type $nnet_type --in-feats 80 --num-enc-blocks $num_layers --enc-hid-units $layer_dim --enc-expand-units $expand_dim --kernel-size $kernel_sizes --dilation $dilation"

opt_opt="--optim.opt-type adam --optim.lr $lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
nnet_name=${nnet_type}_nl${num_layers}ld${layer_dim}_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=70
num_augs=5
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth

#xvector finetuning
# ft_batch_size_1gpu=4
# ft_eff_batch_size=64 # effective batch size
# ft_min_chunk=400
# ft_max_chunk=6000
ft_batch_size_1gpu=16
ft_eff_batch_size=256 # effective batch size
ft_min_chunk=400
ft_max_chunk=800
ft_ipe=0.25
ft_lr=0.005
ft_nnet_num_epochs=40
ft_margin_warmup=3
ft_opt_opt="--optim.opt-type adam --optim.lr $ft_lr --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp"
ft_lrs_opt="--lrsched.lrsch-type exp_lr --lrsched.decay-rate 0.5 --lrsched.decay-steps 8000 --lrsched.hold-steps 40000 --lrsched.min-lr 1e-5 --lrsched.warmup-steps 1000 --lrsched.update-lr-on-opt-step"
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_adam_lr${ft_lr}_b${ft_eff_batch_size}_amp.v2
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0007.pth


#back-end
lda_dim=200
plda_y_dim=150
plda_z_dim=200

plda_data=voxceleb2cat_train_combined
plda_type=splda
