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
lda_dim=150
plda_y_dim=150

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

xvector_dir=exp/xvectors_diar/$nnet_name

plda_name=lda${lda_dim}_${plda_type}y${plda_y_dim}_v1_${plda_data}
plda_dir=exp/be/$nnet_name/$plda_name

plda_adapt_name=${plda_name}_adapt
plda_adapt_dir=exp/be/$nnet_name/$plda_adapt_name

diar_dir=exp/diarization/$nnet_name/${plda_adapt_name}
diar_ahc_dir=$diar_dir/ahc

if [ $stage -le 1 ]; then

    echo "Train PLDA on Voxceleb"
    steps_diar/train_plda_v1.sh --cmd "$train_cmd --mem 64G" \
	--lda-dim $lda_dim \
	--plda-type $plda_type \
	--y-dim $plda_y_dim --z-dim $plda_z_dim \
	--plda-opts "--inter-session" \
	$xvector_dir/$plda_data/xvector.scp \
	data/$plda_data \
	$plda_dir 

fi

if [ $stage -le 2 ]; then

    echo "Adapt PLDA on Dihard dev"
    steps_diar/adapt_plda_v1.sh --cmd "$train_cmd --mem 64G" \
	--plda-type $plda_type \
	--plda-opts "--inter-session --w-mu 1 --w-B 1 --w-W 1" \
	$xvector_dir/dihard2019_dev/xvector.scp \
	data/dihard2019_dev \
	data/dihard2019_dev/diarization.rttm \
	$plda_dir/lda_lnorm.h5 \
	$plda_dir/plda.h5 \
	$plda_adapt_dir 

fi


local/install_dscore.sh

if [ $stage -le 3 ];then
    echo "Apply AHC with PLDA scoring"
    for r in 1 0.5 0.3 0.15
    do
	for threshold in -5 -4 -3 -2 -1 0 #2 3 4 5 6 #-1 0 1 #-3 -2 -1 0 1 2 3 4 5
	do
	    (
		out_dir=${diar_ahc_dir}_pcar${r}_thr${threshold}
		for name in dihard2019_dev dihard2019_eval
		do
    		    steps_diar/eval_ahc_v1.sh \
    			--cmd "$train_cmd --mem 10G" \
    			--ahc-opts "--threshold $threshold --pca-var-r $r --score-hist-dir $out_dir/$name/hist" \
    			data/$name/utt2spk \
    			$xvector_dir/$name/xvector.scp \
    			data/$name/vad.segments \
    			$plda_adapt_dir/lda_lnorm.h5 \
    			$plda_adapt_dir/plda.h5 \
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
		) &
	done
    done
    wait
    exit
fi

plda_dir=$plda_adapt_dir
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
