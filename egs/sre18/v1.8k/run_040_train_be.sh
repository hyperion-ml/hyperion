#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

diar_name=diar1b

net_name=1a

tel_lda_dim=150
vid_lda_dim=200
tel_ncoh=400
vid_ncoh=500
vast_ncoh=120

w_mu1=1
w_B1=0.75
w_W1=0.75
w_mu2=1
w_B2=0.6
w_W2=0.6
num_spks=975

plda_tel_y_dim=125
plda_tel_z_dim=150
plda_vid_y_dim=150
plda_vid_z_dim=200

stage=1

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors/$net_name

coh_vid_data=sitw_sre18_dev_vast_${diar_name}
coh_vast_data=sitw_sre18_dev_vast_${diar_name}
coh_tel_data=sre18_dev_unlabeled
plda_tel_data=sre_tel_combined
plda_tel_type=splda
plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt_v1_a1_mu${w_mu1}B${w_B1}W${w_W1}_a2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}

plda_vid_data=voxceleb_combined
plda_vid_type=splda
plda_vid_label=${plda_vid_type}y${plda_vid_y_dim}_v1

be_tel_name=lda${tel_lda_dim}_${plda_tel_label}_${plda_tel_data}
be_vid_name=lda${vid_lda_dim}_${plda_vid_label}_${plda_vid_data}
be_tel_dir=exp/be/$net_name/$be_tel_name
be_vid_dir=exp/be/$net_name/$be_vid_name


if [ $stage -le 1 ]; then

    steps_be/train_vid_be_v1.sh --cmd "$train_cmd" \
				--lda_dim $vid_lda_dim \
				--plda_type $plda_vid_type \
				--y_dim $plda_vid_y_dim --z_dim $plda_vid_z_dim \
				$xvector_dir/$plda_vid_data/xvector.scp \
				data/$plda_vid_data \
				$xvector_dir/sitw_dev_${diar_name}/xvector.scp \
				data/sitw_dev_${diar_name} \
				$xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp \
				data/sre18_dev_vast_${diar_name} \
				$be_vid_dir &

    steps_be/train_tel_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $tel_lda_dim \
    				--plda_type $plda_tel_type \
    				--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    				--w_mu1 $w_mu1 --w_B1 $w_B1 --w_W1 $w_W1 \
    				--w_mu2 $w_mu2 --w_B2 $w_B2 --w_W2 $w_W2 --num_spks $num_spks \
    				$xvector_dir/$plda_tel_data/xvector.scp \
    				data/$plda_tel_data \
    				$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    				$sre18_dev_meta $be_tel_dir &

    wait

fi

    
exit
