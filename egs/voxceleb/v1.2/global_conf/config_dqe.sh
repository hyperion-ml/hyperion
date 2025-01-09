# ECAPA-TDNN small

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg
nnet_type=resnet1d
nnet_name=${feat_type}_ecapatdnn32x2.v3.1

nnet_s1_base_cfg=conf/train_ecapatdnn32x2_xvec_stage1_v3.1.yaml
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0010.pth

nnet_stage=1 # this will make x-vector extractor to use the netowrk without finetuning.

# back-end
do_plda=false
do_snorm=false
do_qmf=false
do_voxsrc22=false

plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxceleb2cat_train
else
    plda_data=voxceleb2cat_train_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200

