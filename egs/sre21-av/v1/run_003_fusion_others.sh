#!/bin/bash
# Copyright
#                2021   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

echo "This an example of how to fuse audio and visual systems for contrastive systems"

./run_002_fusion_primary.sh \
  --fus-name v1_closed_av_single \
  --audio ../../sre21-av-a/v1.16k/exp/fusion/v2.5.1_fus_pfus0.1_l21e-3_pcal0.05_l21e-4/0 \
  --visual ../../sre21-av-v/v0.2/exp/fusion/v2.4_ptrn0.05_l21e-4/0

./run_002_fusion_primary.sh \
  --fus-name v1_closed_av_contrastive \
  --audio ../../sre21-av-a/v1.16k/exp/fusion/v2.4.1_fus_pfus0.1_l21e-3_pcal0.05_l21e-4/3

./run_002_fusion_primary.sh \
  --fus-name v1_open_av_primary \
  --audio ../../sre21-av-a/v1.16k/exp/fusion/v2.5.1_open_fus_pfus0.1_l21e-3_pcal0.05_l21e-4/4
  
./run_002_fusion_primary.sh \
  --fus-name v1_open_av_contrastive \
  --audio ../../sre21-av-a/v1.16k/exp/fusion/v2.4.1_open_fus_pfus0.1_l21e-3_pcal0.05_l21e-4/4
