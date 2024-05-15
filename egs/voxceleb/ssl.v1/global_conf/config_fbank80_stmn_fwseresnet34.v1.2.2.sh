# ECAPA-TDNN 512x3

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg
nnet_type=resnet
nnet_name=${feat_type}_fwseresnet34_dino.v1.2.2

nnet_s1_base_cfg=conf/train_fwseresnet34_dino_v1.2.2.yaml
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/teacher_model_ep0034.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0038.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0043.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0044.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0046.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0049.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0054.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0058.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0064.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0067.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0071.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0077.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0083.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0088.pth
nnet_s1=$nnet_s1_dir/teacher_model_ep0094.pth

# clustering of dino embeddings
cluster_method=cos_ahc_plda_ahc
cluster_cfg=conf/cluster_lresnet34_v1.2_cos_ahc_plda_ahc.yaml
cluster_name=${cluster_method}
cluster_dir=exp/clustering/$nnet_s1_name/$cluster_name

# plda
plda_cfg=conf/plda.yaml

# finetuning stage 1.1
nnet_ft_s1_1_base_cfg=conf/train_lresnet34_xvec_stage1.1_v1.2.yaml
nnet_ft_s1_1_name=$nnet_name.s1.ft.s1.1
nnet_ft_s1_1_dir=exp/xvector_nnets/$nnet_ft_s1_1_name
nnet_ft_s1_1=$nnet_ft_s1_1_dir/model_ep0030.pth

# finetuning stage 1.2
nnet_ft_s1_2_base_cfg=conf/train_lresnet34_xvec_stage1.2_v1.2.yaml
nnet_ft_s1_2_name=$nnet_name.s1.ft.s1.2
nnet_ft_s1_2_dir=exp/xvector_nnets/$nnet_ft_s1_2_name
nnet_ft_s1_2=$nnet_ft_s1_2_dir/model_ep0070.pth

# clustering of ft embeddings from stage 1.2
cluster_ft_s1_method=cos_ahc
cluster_ft_s1_cfg=conf/cluster_lresnet34_v1.2_ft1_cos_ahc.yaml
cluster_ft_s1_name=${cluster_method_ft_s1_method}
cluster_ft_s1_dir=exp/clustering/$nnet_ft_s1_2_name/$cluster_ft_s1_name

# finetuning stage 2.1
nnet_ft_s2_1_base_cfg=conf/train_lresnet34_xvec_stage1.1_v1.2.yaml
nnet_ft_s2_1_name=$nnet_name.s1.ft.s2.1
nnet_ft_s2_1_dir=exp/xvector_nnets/$nnet_ft_s2_1_name
nnet_ft_s2_1=$nnet_ft_s2_1_dir/model_ep0030.pth

# finetuning stage 2.2
nnet_ft_s2_2_base_cfg=conf/train_lresnet34_xvec_stage1.2_v1.2.yaml
nnet_ft_s2_2_name=$nnet_name.s1.ft.s2.2
nnet_ft_s2_2_dir=exp/xvector_nnets/$nnet_ft_s2_2_name
nnet_ft_s2_2=$nnet_ft_s2_2_dir/model_ep0070.pth

# clustering of ft embeddings from stage 2.2
cluster_ft_s2_method=cos_ahc
cluster_ft_s2_cfg=conf/cluster_lresnet34_v1.2_ft1_cos_ahc.yaml
cluster_ft_s2_name=${cluster_method_ft_s2_method}
cluster_ft_s2_dir=exp/clustering/$nnet_ft_s2_2_name/$cluster_ft_s2_name

