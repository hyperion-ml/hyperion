# LResNet34 x-vector with mixed precision traininga

nnet_name_tag="" # to manage file names for expdir. For example, utilize this when running multiple exps for hyp. para. tuning

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2_train

# x-vector cfg

nnet_type=resnet

resnet_type=lresnet34
batch_size_1gpu=32 # 60,  64 OOM # in dinossl, eff_batch_size, which is not actualy used, is the same as ngpu * batch_size_1gpu since grad_acc_steps is always 1 in dinossl for now. Instead, lr is adjusted linearly proportional to (ngpu * batch_size_1gpu).
eff_batch_size=128 # effective batch size assuming ngpu=4
dropout=0
embed_dim=256
lr=0.005
nnet_num_epochs=70

xvec_train_base_cfg=conf/dinossl_tuning/train_resnet34_xvec_default.yaml
xvec_train_args="--data.train.sampler.batch-size $batch_size_1gpu --model.resnet-type $resnet_type"

# dinossl related (in addition to ones defined in xvec_train_base_cfg): dataset/dataloader, model/loss
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

nnet_name=${feat_type}_${resnet_type}_e${embed_dim}_do${dropout}_adam_b${eff_batch_size}_amp.dinossl.v1
if [[ -n ${nnet_name_tag} ]]; then
    nnet_name=${nnet_name}_${nnet_name_tag}
fi


nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth


# back-end
state_dict_key=model_teacher_state_dict
plda_num_augs=0
plda_data=voxceleb1_test_train
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200


