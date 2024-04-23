# ResNet34

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg
nnet_type=resnet
nnet_name=${feat_type}_lresnet34_dino.v1.1

nnet_s1_base_cfg=conf/train_lresnet34_dino_v1.1.yaml
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/teacher_model_ep0080.pth

# clustering
cluster_method=cos_ahc
cluster_name=${cluster_method}_1
cluster_cfg=conf/ahc.yaml

# plda
plda_cfg=conf/plda.yaml

# finetuning stage 1.1
nnet_ft_s1_1_base_cfg=conf/train_lresnet34_stage1.1_v1.1.yaml
nnet_ft_s1_1_name=$nnet_name.s1.ft.s1.1
nnet_ft_s1_1_dir=exp/xvector_nnets/$nnet_ft_s1_1_name
nnet_ft_s1_1=$nnet_ft_s1_1_dir/model_ep0010.pth

# finetuning stage 1.2
nnet_ft_s1_2_base_cfg=conf/train_lresnet34_stage1.2_v1.1.yaml
nnet_ft_s1_2_name=$nnet_name.s1.ft.s1.2
nnet_ft_s1_2_dir=exp/xvector_nnets/$nnet_ft_s1_2_name
nnet_ft_s1_2=$nnet_ft_s1_2_dir/model_ep0080.pth


# # back-end
# do_plda=false
# # do_snorm=true
# # do_qmf=true
# # do_voxsrc22=true

# plda_aug_config=conf/reverb_noise_aug.yaml
# plda_num_augs=0
# if [ $plda_num_augs -eq 0 ]; then
#     plda_data=voxceleb2cat_train
# else
#     plda_data=voxceleb2cat_train_augx${plda_num_augs}
# fi
# plda_type=splda
# lda_dim=200
# plda_y_dim=150
# plda_z_dim=200

