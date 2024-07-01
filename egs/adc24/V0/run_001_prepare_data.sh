#!/bin/bash
# Copyright
#                2024   Johns Hopkins University (Author: Mohammed Akram Khelfi)
#
# 

. ./cmd.sh
. ./path.sh

conda activate adc24
python -V

set -e 

config_file=default_config.sh
stage=1

. parse_options.sh || exit 1;

. datapath.sh

if [ $stage -le 1 ];then
      hyp_utils/conda_env.sh \
	  local/prepare_adi17.py \
	  --corpus-dir $adi_root \
	  --output-dir data/adi17 \
	  --map-langs-to-lre-codes --target-fs 16000
fi



