# LResNet34 x-vector with mixed precision training
# Some variables defined here are just used to define nnet_name, which is better to be fixed

nnet_name_tag="" # to manage file names for expdir. For example, utilize this when running multiple exps for hyp. para. tuning

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn # just to define nnet_name. When changing this, fix the part in ${xvec_train_base_cfg} too to be applied in the actual training setup

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2_train

# x-vector cfg

nnet_type=resnet

resnet_type=lresnet34
batch_size_1gpu=48 # 52:64:4 OOM
ngpu=2
eff_batch_size=`expr $batch_size_1gpu \* $ngpu` # In dinossl, eff_batch_size is the same as ngpu * batch_size_1gpu since grad_acc_steps is always 1 in dinossl for now. Thus, when eff_batch_size changes, instead of changing grad_acc_steps w/ a fixed lr, lr (base_value in cosine_scheduler, to be exact) is adjusted linearly proportional to eff_batch_size where the base value is 0.005 as as a default w/ eff_batch_size=256. For example, if eff_batch_size=128, the base value is 0.0025 in dinossl. eff_batch_size is calculated in python scripts but one here is to compose nnet_name below. # just to define nnet_name. When changing this, fix the part in ${xvec_train_base_cfg} too to be applied in the actual training setup
dropout=0 # just to define nnet_name. When changing this, fix the part in ${xvec_train_base_cfg} too to be applied in the actual training setup
embed_dim=256 # just to define nnet_name. When changing this, fix the part in ${xvec_train_base_cfg} too to be applied in the actual training setup

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

nnet_name=${feat_type}_${resnet_type}_e${embed_dim}_do${dropout}_b${eff_batch_size}_amp.dinossl.v1
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


