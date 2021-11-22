# LResNet34 x-vector with mixed precision training
# Edited from
# ../v1.1/global_conf/config_fbank80_stmn_lresnet34_arcs30m0.3_adam_lr0.05_amp.dinossl_ebs256_keep_classif_net.v1.sh
nnet_name_tag="" # to manage file names for expdir. For example, utilize this when running multiple exps for hyp. para. tuning

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml


# x-vector training
nnet_data=voxceleb2_train
nnet_num_augs=1
aug_opt="--train-aug-cfg conf/reverb_noise_aug.yaml --val-aug-cfg conf/reverb_noise_aug.yaml"

batch_size_1gpu=64 # effective batch size is 128 when using 2 gpus. IF "--lrsched.lrsch-type dinossl" is given, lr is linearly scaled with the batch size in the training code.
ngpu=2
eff_batch_size=$((batch_size_1gpu*ngpu))
ipe=$nnet_num_augs
min_chunk=2
max_chunk=2
lr=0.005 # This is lr for a reference effective batch size of 256. For example, if effective batch size = 128 (256 * 0.5), lr is adjusted to 0.005 * 0.5 in the training code automatically IF "--lrsched.lrsch-type dinossl" is given)

nnet_type=lresnet34 # light resnet
dropout=0
embed_dim=256 # x-vector embedding size when either "dinossl_fclikeimage" or "dinossl_keep_classif_net" is True

nnet_opt="--resnet-type $nnet_type --in-feats 80 --in-channels 1 --in-kernel-size 3 --in-stride 1 --no-maxpool"

opt_opt="--optim.opt-type adam --optim.beta1 0.9 --optim.beta2 0.95 --optim.weight-decay 1e-5 --optim.amsgrad --use-amp --optim.dinossl_style True"
lrs_opt="--lrsched.lrsch-type dinossl --lrsched.dinossl_lr $lr --lrsched.dinossl_min_lr 1e-6 --lrsched.dinossl_warmup_epochs 10 --lrsched.dinossl_weight_decay 1e-4 --lrsched.dinossl_weight_decay_end 1e-4 --lrsched.dinossl_momentum_teacher 0.996"

# dinossl related: dataset/dataloader, model/loss
dinossl=true
dinossl_keep_classif_net=true
## dino-head
dinossl_out_dim=65536
dinossl_use_bn_in_head=false
dinossl_norm_last_layer=true
## data-augmentation
dinossl_local_crops_number=4
## teacher temperature
dinossl_warmup_teacher_temp=0.04
dinossl_teacher_temp=0.04
dinossl_warmup_teacher_temp_epochs=0
## chunk sampling related
dinossl_chunk_len_mult=2 # a factor by which long chunk length increases from short chunk length. The short chunk length is determined randomly between min_chunk and max_chunk set above
if [[ ${dinossl_keep_classif_net} == true ]]; then
    nnet_name=${feat_type}_${nnet_type}_e${embed_dim}_do${dropout}_adam_b${eff_batch_size}_amp.dinossl_keep_classif_net.v1
elif [[ ${dinossl_fclikeimage} == true ]]; then
    nnet_name=${feat_type}_${nnet_type}_e${embed_dim}_do${dropout}_adam_b${eff_batch_size}_amp.dinossl_fclikeimage.v1
else # when only ${dinossl} == true. embed_dim will be calculated in the training code as the dim of pooling output
    nnet_name=${feat_type}_${nnet_type}_do${dropout}_adam_b${eff_batch_size}_amp.dinossl.v1
fi

if [[ -n ${nnet_name_tag} ]]; then
    nnet_name=${nnet_name}_${nnet_name_tag}
fi
nnet_num_epochs=70
nnet_dir=exp/xvector_nnets/${nnet_name} # expdir (outdir)


# back-end
nnet=$nnet_dir/model_ep0059.pth # ep0059 corresponds to ep0070 of cat w/ nnet_num_augs=6 case. Simply change to ep0070 if you want use the later model.
state_dict_key=model_teacher_state_dict
plda_num_augs=0
plda_data=voxceleb1_test_train
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200

