# ECAPA-TDNN small

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data_1000=voxceleb2cat_1000
nnet_data=voxceleb2cat_500
full_dataset=voxceleb2cat_full

# x-vector cfg
nnet_type=resnet1d
nnet_name=${feat_type}_ecapatdnn512x3.v3.0

alpha_min=1
alpha_max=$alpha_min
#alpha_max=2.55285052685111

config=6
nnet_s1_base_cfg=conf/train_ecapatdnn512x3_xvec_stage3_v3.0.yaml
nnet_s1_name=$nnet_name.s1
#nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
#nnet_s1=${nnet_s1_dir}_var_length_c${config}/model_ep0160.pth
nnet_s1_dir=exp/xvector_nnets/baseline/${nnet_s1_name}_$nnet_data
nnet_s1=${nnet_s1_dir}/model_ep0040.pth

nnet_s2_base_cfg=conf/train_ecapatdnn512x3_xvec_stage2_v3.0.yaml
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0030.pth
nnet_s2=$nnet_s2_dir/swa_model_ep0036.pth

# back-end
do_plda=false
do_snorm=true
do_qmf=true
do_voxsrc22=fasle

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

# clustering 
cluster_method=kmeans
cluster_cfg=conf/cluster_ecapatdnn512x3_v1.2_ft1_cos_ahc.yaml
cluster_name=${cluster_method}
cluster_dir=exp/clustering/$nnet_s1_name/$cluster_name