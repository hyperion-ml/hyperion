#!/bin/bash
# Copyright
#                2024   Johns Hopkins University (Author: Mohammed Akram Khelfi)
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

# if [ $stage -le 1 ]; then
#   # This script preprocess audio for x-vector training
#   for name in train
#   do
#     steps_xvec/preprocess_audios_for_nnet_train.sh \
#       --nj 40 --cmd "$train_cmd" \
#       --storage_name adc24-adi17-${name}-$(date +'%m_%d_%H_%M') --use-bin-vad false \
#       data/adi17/${name} data/adi17/${name}_proc_audio_no_sil exp/adi17/${name}_proc_audio_no_sil
    
#     utils/fix_data_dir.sh data/adi17/${name}_proc_audio_no_sil
#   done
# fi

if [ $stage -le 3 ]; then
  # Now, we remove files with less than 3s
  hyp_utils/remove_short_audios.sh --min-len 3 data/adi17/train_proc_audio_no_sil
  hyp_utils/remove_short_audios.sh --min-len 3 data/adi17/test_proc_audio_no_sil
  hyp_utils/remove_short_audios.sh --min-len 3 data/adi17/dev_proc_audio_no_sil
  
fi



exit