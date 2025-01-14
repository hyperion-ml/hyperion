#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

# hyperion-dataset remove_classes_few_segments \
# 		  --dataset data/voxceleb2cat_100_xvector_val \
# 		  --class-name speaker --min-segs 1

# trigger_dir=data/triggers/click/new/wav

# baseline=data/triggers/click/trimmed/best/mixkit-camera-shutter-click-1133.wav

# mkdir -p $trigger_dir/trimmed
# hyperion-dataset adjust_length\
#                    --input-dir $trigger_dir \
#                    --output-dir $trigger_dir/trimmed \
#                    --target-path $baseline \

# mkdir -p $trigger_dir/trimmed_n_normalized
# hyperion-dataset adjust_vols\
#                    --input-dir $trigger_dir/trimmed \
#                    --output-dir $trigger_dir/trimmed_n_normalized \
#                    --target-path $baseline \


