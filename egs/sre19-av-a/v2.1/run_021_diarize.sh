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
	$plda_dir &
    wait
fi


if [ $stage -le 2 ];then
    echo "Apply AHC with PLDA scoring"
    for r in 1 #0.5 0.3 0.15
    do
	for threshold in -7 -5 -3 -1 0 
	do
	    
	    out_dir=${diar_ahc_dir}_pcar${r}_thr${threshold}
	    for name in sitw_dev_test sitw_eval_test \
		sre18_eval_test_vast sre18_dev_test_vast \
		sre19_av_a_dev_test sre19_av_a_eval_test \
		janus_dev_test_core janus_eval_test_core
	    do
    		steps_diar/eval_ahc_v1.sh \
    		    --cmd "$train_cmd --mem 4G" --nj 20 \
    		    --ahc-opts "--threshold $threshold --pca-var-r $r --score-hist-dir $out_dir/$name/hist" \
    		    data/$name/utt2spk \
    		    $xvector_dir/$name/xvector.scp \
    		    scp:data/$name/vad.scp \
    		    $plda_dir/lda_lnorm.h5 \
    		    $plda_dir/plda.h5 \
    		    $out_dir/$name &
	    done
	done
    done
    wait

fi
exit

if [ $stage -le 2 ];then
    echo "Apply AHC with PLDA scoring"
    for r in 1 0.5 0.3 0.15
    do
	for threshold in -3 -2 -1 0 #2 3 4 5 6 #-1 0 1 #-3 -2 -1 0 1 2 3 4 5
	do
	    out_dir=${diar_ahc_dir}_pcar${r}_unsupcal_thr${threshold}
	    for name in dihard2019_dev dihard2019_eval
	    do
    		steps_diar/eval_ahc_v1.sh \
    		    --cmd "$train_cmd --mem 10G" \
    		    --ahc-opts "--threshold $threshold --pca-var-r $r --do-unsup-cal --score-hist-dir $out_dir/$name/hist" \
    		    data/$name/utt2spk \
    		    $xvector_dir/$name/xvector.scp \
    		    data/$name/vad.segments \
    		    $plda_dir/lda_lnorm.h5 \
    		    $plda_dir/plda.h5 \
    		    $out_dir/$name
	    done
	    local/dscore_dihard2019_allconds.sh \
		$dihard2019_dev/data/single_channel \
		$out_dir/dihard2019_dev/rttm \
		$out_dir/dihard2019_dev
	    local/dscore_dihard2019_allconds.sh \
		$dihard2019_eval/data/single_channel \
		$out_dir/dihard2019_eval/rttm \
		$out_dir/dihard2019_eval
	done
    done

fi

exit
