#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

xvector_dir=exp/xvectors/$nnet_name
be_tel_dir=exp/be/$nnet_name/$be_tel_name
be_vid_dir=exp/be/$nnet_name/$be_vid_name


if [ $stage -le 1 ]; then


    steps_be/train_tel_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_tel_dim \
    				--plda_type $plda_tel_type \
    				--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    				--w_mu1 $w_mu1 --w_B1 $w_B1 --w_W1 $w_W1 \
    				--w_mu2 $w_mu2 --w_B2 $w_B2 --w_W2 $w_W2 --num_spks $num_spks \
    				$xvector_dir/$plda_tel_data/xvector.scp \
    				data/$plda_tel_data \
    				$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    				$sre18_dev_meta $be_tel_dir &

    
    steps_be/train_vid_be_v1.sh --cmd "$train_cmd" \
				--lda_dim $lda_vid_dim \
				--plda_type $plda_vid_type \
				--y_dim $plda_vid_y_dim --z_dim $plda_vid_z_dim \
				$xvector_dir/$plda_vid_data/xvector.scp \
				data/$plda_vid_data \
				$xvector_dir/sitw_dev1s_${diar_name}/xvector.scp \
				data/sitw_dev1s_${diar_name} \
				$xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
				data/sre18_dev_vast_${diar_name} \
				$be_vid_dir &


    wait

fi

    
exit
