# ConvNext2d Atto size

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
#vad_config=conf/vad_16k.yaml
# we do

# x-vector training 
nnet_train_data=asvspoof2024_train
nnet_val_data=asvspoof2024_dev

# x-vector cfg
nnet_type=convnext2d
nnet_name=${feat_type}_convnext2d_atto.v1.0

nnet_s1_base_cfg=conf/train_convnext2d_atto_xvec_stage1_v1.0.yaml
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0030.pth

# nnet_s2_base_cfg=conf/train_convnext2d_atto_xvec_stage2_v3.1.yaml
# nnet_s2_name=${nnet_name}.s2
# nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
# nnet_s2=$nnet_s2_dir/swa_model_ep0016.pth

# # back-end
# do_plda=false
# do_snorm=true
# do_qmf=true
# do_voxsrc22=true

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

