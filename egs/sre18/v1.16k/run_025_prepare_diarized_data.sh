#!/bin/bash
# Copyright     2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e
vaddiardir=`pwd`/vad_diar

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ $stage -le 1 ]; then
    # Create datasets with diarization
    for name in sitw_dev_test sitw_eval_test sre18_dev_test_vast sre18_eval_test_vast
    do
	rttm=$rttm_dir/$name/plda_scores_t${diar_thr}/rttm
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur data/$name $rttm data/${name}_${diar_name} $vaddiardir
    done

fi

if [ $stage -le 2 ]; then
    mkdir -p data/sitw_eval_test_${diar_name}/trials data/sitw_dev_test_${diar_name}/trials
    for file in assist-core.lst assist-multi.lst core-core.lst core-multi.lst
    do
	local/make_diar_trials.sh data/sitw_dev_test_${diar_name}/orig2utt data/sitw_dev_test/trials/$file data/sitw_dev_test_${diar_name}/trials/$file
	local/make_diar_trials.sh data/sitw_eval_test_${diar_name}/orig2utt data/sitw_eval_test/trials/$file data/sitw_eval_test_${diar_name}/trials/$file 
    done
    local/make_diar_trials.sh data/sre18_dev_test_vast_${diar_name}/orig2utt data/sre18_dev_test_vast/trials data/sre18_dev_test_vast_${diar_name}/trials
    local/make_diar_trials.sh data/sre18_eval_test_vast_${diar_name}/orig2utt data/sre18_eval_test_vast/trials data/sre18_eval_test_vast_${diar_name}/trials

fi

exit
