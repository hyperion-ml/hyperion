# Wav2vec2 base trained on 960h LibriSpeech + ECAPA-TDNN 512x2

# hugging face model
hf_model_name=wav2vec2base

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=voxceleb2cat_train

# x-vector cfg

nnet_type=hf_wav2vec2resnet1d

batch_size_1gpu=32
eff_batch_size=512 # effective batch size
dropout=0
embed_dim=256
lr=0.05
s=30
margin_warmup=20
margin=0.3
nnet_num_epochs=70


lr=0.001
#lr=0.005
xvec_train_base_cfg=conf/train_wav2vec2base_ecapatdnn512x2_default.yaml
xvec_train_args="--data.train.sampler.batch-size $batch_size_1gpu --trainer.optim.lr $lr --trainer.lrsched.warmup-steps 20000 --trainer.lrsched.hold-steps 20000 --trainer.lrsched.min-lr 1e-6 --trainer.epochs 75"

nnet_name=${hf_model_name}_ecapatdnn512x2_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v3 #v1

nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0060.pth
nnet=$nnet_dir/swa_model_ep0076.pth
nnet=$nnet_dir/model_ep0060.pth

# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=6
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxceleb2cat_train
else
    plda_data=voxceleb2cat_train_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200

