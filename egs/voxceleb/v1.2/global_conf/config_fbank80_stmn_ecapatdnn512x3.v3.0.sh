# ---------------------------------------------------------------------------
# ECAPA-TDNN (512x3) baseline configuration
#
# This file is sourced by the recipe stage scripts.  It sets the acoustic
# feature extractor, VAD definitions, neural net checkpoints, and back-end
# switches used throughout the pipeline.  Feel free to clone this template and
# adjust paths / toggles when running custom experiments.
# ---------------------------------------------------------------------------

# Acoustic feature front-end: short-time mean-normalised 80-dim filterbanks.
# ``feat_config`` points to the YAML used by the data-prep scripts and the
# training recipe, while ``feat_type`` becomes part of directory names.

# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

# Voice activity detection model used when generating training/eval segments.
vad_config=conf/vad_16k.yaml

# Training data list used when fitting the x-vector/ECAPA model.
nnet_data=voxceleb2cat_train

# Neural-network naming.  ``nnet_type`` selects the python module to import
# while ``nnet_name`` is used to create checkpoint directories.
nnet_type=resnet1d
nnet_name=${feat_type}_ecapatdnn512x3.v3.0

# Stage-1 (pretraining) configuration and checkpoint locations.
nnet_s1_base_cfg=conf/train_ecapatdnn512x3_xvec_stage1_v3.0.yaml
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0040.pth

# Stage-2 (fine-tuning) configuration and checkpoint locations.
nnet_s2_base_cfg=conf/train_ecapatdnn512x3_xvec_stage2_v3.0.yaml
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0030.pth
# If you prefer the stochastic-weight-averaged model, uncomment the line
# below (or comment out the previous assignment).
nnet_s2=$nnet_s2_dir/swa_model_ep0036.pth

# Back-end switches controlling PLDA/QMF/S-Norm evaluation stages.
do_plda=false      # set to true to train and evaluate a PLDA backend
do_snorm=true      # score normalisation using the S-Norm recipe
do_qmf=true        # Quality Measure Function based score calibration
do_voxsrc22=true   # Whether to run the VoxSRC'22 evaluation track

# Data augmentation used when training PLDA back-ends.  When ``plda_num_augs``
# is zero we fall back to the clean VoxCeleb2 cat set, otherwise we load the
# augmented list created by the prep scripts.
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
