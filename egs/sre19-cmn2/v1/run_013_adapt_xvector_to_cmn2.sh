#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

nnet_data=adapt_combined
nnet_num_epochs=3
stage=6
if [ $stage -le 8 ]; then
    steps_kaldi_xvec/run_xvector_${nnet_vers}_adapt.sh --stage $stage --train-stage 0 --num_epochs $nnet_num_epochs \
    	--storage_name sre19-cm2-v1-$(date +'%m_%d_%H_%M') --nodes "bc" \
	--lr 0.0001 --final-lr 0.000001 --num-repeats 9 --batch-size 162 \
    	--data data/${nnet_data}_no_sil \
	--init-nnet-dir $nnet_dir \
	--nnet-dir ${nnet_dir}_adapt_cmn2 \
    	--egs-dir ${nnet_dir}_adapt_cmn2/egs

fi

exit
