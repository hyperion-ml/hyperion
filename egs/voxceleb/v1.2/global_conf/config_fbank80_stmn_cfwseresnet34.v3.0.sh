# ---------------------------------------------------------------------------
# Channel-Frequency-Wise SE-ResNet34 (v3.0) baseline configuration
#
# This file is sourced by the VoxCeleb v1.2 recipe. Update the paths or
# toggles below to spin up custom experiments. The sections configure the
# acoustic front-end, VAD definition, neural-net checkpoints, and the back-end
# evaluation switches.
# ---------------------------------------------------------------------------

# Acoustic feature front-end shared across the pipeline
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

# Voice activity detection definition used during data prep
vad_config=conf/vad_16k.yaml

# Training data partition for fitting the x-vector model

nnet_data=voxceleb2cat_train

# Neural network type, naming, and checkpoint locations
nnet_type=resnet
nnet_name=${feat_type}_cfwseresnet34.v3.0

# Stage-1 configuration (pretraining)
nnet_s1_base_cfg=conf/train_cfwseresnet34_xvec_stage1_v3.0.yaml
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0035.pth

# Stage-2 configuration (fine-tuning / SWA)
nnet_s2_base_cfg=conf/train_cfwseresnet34_xvec_stage2_v3.0.yaml
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/swa_model_ep0016.pth

# Back-end toggles (PLDA, score normalisation, calibration)
do_plda=false
do_snorm=true
do_qmf=true
do_voxsrc22=true

# Data augmentation recipe used when training PLDA back-ends
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
