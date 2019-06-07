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

stage=1
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file


#Train embeddings
if [ $stage -le 1 ]; then
    train_args="--net-vers $nnet_vers --init_s $init_s --lr $lr --p-drop $p_drop \
                --x_dim $nnet_x_dim --y_dim $nnet_y_dim \
		--num_layers_y $nnet_num_layers_y \
		--num_layers_t $nnet_num_layers_t \
	        --h_y_dim $nnet_h_y_dim --h_t_dim $nnet_h_t_dim \
                --lr_decay $lr_decay --ipe $ipe \
		--act $nnet_act --batch_size $batch_size --epochs $nnet_num_epochs \
                --min_frames $min_frames --max_frames $max_frames --lr_patience $lr_patience --patience $patience"
    steps_embed/train_embed_gen_v2.sh --cmd "$cuda_cmd" \
				      $train_args \
				      data/${nnet_data}_no_sil/feats.scp \
				      data/lists_embed/${nnet_data} \
				      $nnet_dir
fi

