#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

net_vers=1a
net_name=1a.1

x_dim=23
y_dim=1500
h_y_dim=512
h_t_dim=512
num_layers_y=5
num_layers_t=3
act=relu

min_frames=200
max_frames=400
batch_size=256
ipe=1
#lr=0.000125
lr=0.00025
##lr=0.0005
###lr=0.00125
lr_decay=0
init_s=0.1
p_drop=0.1
lr_patience=3
patience=10
num_epochs=200

nnet_dir=exp/xvector_nnet/$net_name

stage=1

. parse_options.sh || exit 1;


#Train embeddings
if [ $stage -le 1 ]; then
    train_args="--net-vers $net_vers --init_s $init_s --lr $lr --p-drop $p_drop \
                --x_dim $x_dim --y_dim $y_dim \
		--num_layers_y $num_layers_y \
		--num_layers_t $num_layers_t \
	        --h_y_dim $h_y_dim --h_t_dim $h_t_dim \
                --lr_decay $lr_decay --ipe $ipe \
		--act $act --batch_size $batch_size --epochs $num_epochs \
                --min_frames $min_frames --max_frames $max_frames --lr_patience $lr_patience --patience $patience"
    steps_embed/train_embed_gen_v2.sh --cmd "$cuda_cmd" \
				      $train_args \
				      data/train_combined_no_sil/feats.scp \
				      data/lists_embed/train_combined \
				      $nnet_dir
    exit    

fi

