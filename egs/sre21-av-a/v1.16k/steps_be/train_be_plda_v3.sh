#!/bin/bash
#
# Copyright 2020 Johns Hopkins University (Jesus Villalba)
#           
# Apache 2.0.
set -e

cmd=run.pl
lda_dim=150
plda_type=frplda
y_dim=100
z_dim=150
pca_var_r=1
r_mu=25
r_s=100
w_mu1=0.5
w_B1=0.5
w_W1=0.5
w_mu2=0.5
w_B2=0.5
w_W2=0.5

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <vector-dir> <model-dir>"
  echo "e.g.: $0 exp/xvectors/voxceleb/xvector.scp data/voxceleb exp/be/train_plda"
fi

vector_dir=$1
output_dir=$2

mkdir -p $output_dir/log

$cmd $output_dir/log/train_be.log \
     hyp_utils/conda_env.sh \
     steps_be/train-be-plda-v3.py \
     --v-dir $vector_dir \
     --vox-name voxcelebcat \
     --sre-name sre_alllangs \
     --sre21-enr-name sre21_audio_dev_enroll \
     --sre21-test-name sre21_audio_dev_test \
     --r-mu $r_mu --r-s $r_s --pca-var-r $pca_var_r \
     --plda-type $plda_type \
     --y-dim $y_dim --z-dim $z_dim \
     --w-mu1 $w_mu1 --w-B1 $w_B1 --w-W1 $w_W1 \
     --w-mu2 $w_mu2 --w-B2 $w_B2 --w-W2 $w_W2 \
     --output-path $output_dir

