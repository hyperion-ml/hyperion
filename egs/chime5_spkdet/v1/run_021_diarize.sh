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

plda_data=$diar_plda_data
plda_type=$diar_plda_type
lda_dim=$diar_lda_dim
plda_y_dim=$diar_plda_y_dim
plda_z_dim=$diar_plda_z_dim

xvector_dir=exp/xvectors_diar/$nnet_name

plda_name=lda${lda_dim}_${plda_type}y${plda_y_dim}_v1_${plda_data}
plda_dir=exp/be_diar/$nnet_name/$plda_name

diar_dir=exp/diarization/$nnet_name/${plda_name}
diar_ahc_dir=$diar_dir/ahc

if [ $stage -le 1 ]; then

    echo "Train PLDA on Voxceleb"
    steps_diar/train_plda_v1.sh --cmd "$train_cmd --mem 32G" \
	--lda_dim $lda_dim \
	--plda_type $plda_type \
	--y_dim $plda_y_dim --z_dim $plda_z_dim \
	--plda-opts "--inter-session" \
	$xvector_dir/$plda_data/xvector.scp \
	data/$plda_data \
	$plda_dir 

fi


if [ $stage -le 2 ];then
    echo "Apply AHC with PLDA scoring"
    for r in 1 #0.5 0.3 0.15
    do
	for threshold in -10 #-7 -5 -3 -1 0 
	do
	    
	    out_dir=${diar_ahc_dir}_pcar${r}_thr${threshold}
	    for xvec_name in chime5_spkdet_test
	    do
		for name in chime5_spkdet_test
		do
    		    steps_diar/eval_ahc_v1.sh \
    			--cmd "$train_cmd --mem 4G" --nj 20 \
    			--ahc-opts "--threshold $threshold --pca-var-r $r --score-hist-dir $out_dir/$name/hist" \
    			data/$name/utt2spk \
    			$xvector_dir/$xvec_name/xvector.scp \
    			scp:data/$name/vad.scp \
    			$plda_dir/lda_lnorm.h5 \
    			$plda_dir/plda.h5 \
    			$out_dir/$name &
		done
	    done
	done
    done
    wait

fi
exit

