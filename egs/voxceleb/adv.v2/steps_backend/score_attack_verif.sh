#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-root> <score-dir>"
  exit 1;
fi

set -e

data_dir=$1
score_dir=$2

echo "Voxceleb2 Test Attack-Verif"
key=$data_dir/trials
hyp_utils/conda_env.sh local/score_dcf.py \
    --key-file $key \
    --score-file $score_dir/attack_verif_scores \
    --output-path $score_dir/attack_verif_all &
for cond in known unknown
do
    #Compute performance
    key=$data_dir/trials_$cond
    hyp_utils/conda_env.sh local/score_dcf.py --key-file $key \
	--score-file $score_dir/attack_verif_scores \
	--output-path $score_dir/attack_verif_${cond} &

done
wait

