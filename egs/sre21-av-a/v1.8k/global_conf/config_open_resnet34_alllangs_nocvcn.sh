# LResNet34 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank64_stmn_8k.yaml
feat_type=fbank64_stmn

#vad
vad_config=conf/vad_8k.yaml

# x-vector training 
ft_nnet_name=fbank64_stmn_resnet34_eina_hln_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v3
ft_nnet_dir=exp/xvector_nnets_open/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0045.pth

#fbank64_stmn_tseres2net50w26s4_r256_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b510_amp.v1.alllangs_nocveng.ft_10_15_arcm0.3_sgdcos_lr0.05_b128_amp.v2
#fbank64_stmn_tseres2net50w26s4_r256_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_16_arcm0.3_sgdcos_lr0.05_b128_amp.v2


# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
plda_type=splda
# lda_dim=200
# plda_y_dim=150
# plda_z_dim=200

