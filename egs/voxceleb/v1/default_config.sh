#Default configuration parameters for the experiment

#xvector training 

#xvector training 
nnet_data=voxceleb_combined
nnet_type=tdnn
batch_size_1gpu=256
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.1
dropout=0.1
embed_dim=256
s=30
margin_warmup=3
margin=0.3
num_layers=9
layer_dim=512
expand_dim=1536
dilation="1 2 3 4 1 1 1 1 1"
kernel_sizes="5 3 3 3 1 1 1 1 1"
opt_opt="--opt-optimizer sgd --opt-lr $lr --opt-momentum 0.9 --opt-weight-decay 1e-5"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 4000 --lrsch-hold-steps 20000 --lrsch-min-lr 1e-6 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-batch"
nnet_name=tdnn_nl${num_layers}ld${layer_dim}e${embed_dim}_arc${margin}_do${dropout}_sgd_lr${lr}_b${eff_batch_size}.v1
nnet_num_epochs=200
num_augs=5
nnet_dir=exp/xvector_nnets/$nnet_name

#diarization back-end
lda_diar_dim=120
plda_diar_data=voxceleb
be_diar_name=lda${lda_diar_dim}_plda_${plda_diar_data}

diar_thr=-0.9
min_dur=10
rttm_dir=./exp/diarization/$nnet_name/$be_diar_name
diar_name=diar${nnet_name}_thr${diar_thr}

#spk det back-end
lda_vid_dim=200
ncoh_vid=500
ncoh_vast=120

plda_vid_y_dim=150
plda_vid_z_dim=200

coh_vid_data=sitw_sre18_dev_vast_${diar_name}
coh_vast_data=sitw_sre18_dev_vast_${diar_name}

plda_vid_data=voxceleb_combined
plda_vid_type=splda
plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v1

be_vid_name=lda${lda_vid_dim}_${plda_vid_label}_${plda_vid_data}
