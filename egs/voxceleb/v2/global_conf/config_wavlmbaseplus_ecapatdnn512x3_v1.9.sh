# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wavlmbaseplus

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg

nnet_type=hf_wavlm2resnet1d

xvec_train_base_cfg=conf/train_wavlmbaseplus_ecapatdnn512x3_phase1_default.yaml
xvec_train_args="--model.xvector.margin-warmup-epochs 5 --trainer.lrsched.decay-steps 3150 --trainer.lrsched.warmup-steps 1500 --trainer.lrsched.hold-steps 1500 --trainer.epochs 45 --model.feat-fusion-method weighted-avg --model.feat-fusion-start 2 --model.xvector.intertop-margin 0.1"

nnet_name=${hf_model_name}_ecapatdnn512x3_v1.9

nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0045.pth

# back-end
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

