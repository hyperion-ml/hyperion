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

xvector_dir=exp/xvectors/$nnet_name
vaddir_diar=./exp/vaddiar/${diar_name}

if [ $stage -le 1 ]; then
    # Create datasets with diarization
    for name in sitw_dev_test sitw_eval_test \
	sre18_eval_test_vast sre18_dev_test_vast \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_test_core janus_eval_test_core
    do
	rttm=$rttm_dir/$name/plda_scores_t${diar_thr}/rttm
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur_spkdet_subsegs data/$name $rttm data/${name}_${diar_name} $vaddir_diar
    done

fi

if [ $stage -le 2 ]; then
    # Extract xvectors for diarization data

    for name0 in sitw_dev_test sitw_eval_test \
	sre18_eval_test_vast sre18_dev_test_vast \
	sre19_av_a_dev_test sre19_av_a_eval_test \
	janus_dev_test_core janus_eval_test_core
    do
	name=${name0}_${diar_name}
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done
fi

if [ $stage -le 3 ]; then
    # combine datasets    
    utils/combine_data.sh data/sitw_dev_${diar_name} \
	data/sitw_dev_enroll data/sitw_dev_test_${diar_name}
    mkdir -p $xvector_dir/sitw_dev_${diar_name}
    cat $xvector_dir/sitw_dev_{enroll,test_${diar_name}}/xvector.scp \
	> $xvector_dir/sitw_dev_${diar_name}/xvector.scp

    mkdir -p $xvector_dir/sitw_eval_${diar_name}
    cat $xvector_dir/sitw_eval_{enroll,test_${diar_name}}/xvector.scp \
	> $xvector_dir/sitw_eval_${diar_name}/xvector.scp
    
    utils/combine_data.sh data/sre18_dev_vast_${diar_name} \
	data/sre18_dev_enroll_vast data/sre18_dev_test_vast_${diar_name} 
    mkdir -p $xvector_dir/sre18_dev_vast_${diar_name}
    cat $xvector_dir/sre18_dev_{enroll_vast,test_vast_${diar_name}}/xvector.scp \
	> $xvector_dir/sre18_dev_vast_${diar_name}/xvector.scp

    utils/combine_data.sh data/sre18_eval_vast_${diar_name} \
	data/sre18_eval_enroll_vast data/sre18_eval_test_vast_${diar_name} 
    mkdir -p $xvector_dir/sre18_eval_vast_${diar_name}
    cat $xvector_dir/sre18_eval_{enroll_vast,test_vast_${diar_name}}/xvector.scp \
	> $xvector_dir/sre18_eval_vast_${diar_name}/xvector.scp

    utils/combine_data.sh data/sre19_av_a_dev_${diar_name} \
	data/sre19_av_a_dev_enroll data/sre19_av_a_dev_test_${diar_name} 
    mkdir -p $xvector_dir/sre19_av_a_dev_${diar_name}
    cat $xvector_dir/sre19_av_a_dev_{enroll,test_${diar_name}}/xvector.scp \
	> $xvector_dir/sre19_av_a_dev_${diar_name}/xvector.scp

    utils/combine_data.sh data/sre19_av_a_eval_${diar_name} \
	data/sre19_av_a_eval_enroll data/sre19_av_a_eval_test_${diar_name} 
    mkdir -p $xvector_dir/sre19_av_a_eval_${diar_name}
    cat $xvector_dir/sre19_av_a_eval_{enroll,test_${diar_name}}/xvector.scp \
	> $xvector_dir/sre19_av_a_eval_${diar_name}/xvector.scp

    utils/combine_data.sh data/janus_dev_core_${diar_name} \
	data/janus_dev_enroll data/janus_dev_test_core_${diar_name}
    mkdir -p $xvector_dir/janus_dev_core_${diar_name}
    cat $xvector_dir/janus_dev_{enroll,test_core_${diar_name}}/xvector.scp > \
	$xvector_dir/janus_dev_core_${diar_name}/xvector.scp

    utils/combine_data.sh data/janus_eval_core_${diar_name} \
	data/janus_eval_enroll data/janus_eval_test_core_${diar_name}
    mkdir -p $xvector_dir/janus_eval_core_${diar_name}
    cat $xvector_dir/janus_eval_{enroll,test_core_${diar_name}}/xvector.scp \
	> $xvector_dir/janus_eval_core_${diar_name}/xvector.scp

fi

if [ $stage -le 4 ]; then
    utils/combine_data.sh data/sitw_sre18_dev_vast_${diar_name} \
	data/sitw_dev_${diar_name} data/sre18_dev_vast_${diar_name}
    mkdir -p $xvector_dir/sitw_sre18_dev_vast_${diar_name}
    cat $xvector_dir/{sitw_dev_${diar_name},sre18_dev_vast_${diar_name}}/xvector.scp \
	> $xvector_dir/sitw_sre18_dev_vast_${diar_name}/xvector.scp
fi

    
exit
