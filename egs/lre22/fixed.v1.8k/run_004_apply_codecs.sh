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

if [ $stage -le 1 ];then

  for data in voxlingua107 lre17_dev_afv lre17_eval_afv
  do
      hyp_utils/conda_env.sh \
	local/apply_tel_codecs_to_kaldi_datadir.py \
	--input-dir data/$data \
	--output-dir data/${data}_codecs
  done

fi
