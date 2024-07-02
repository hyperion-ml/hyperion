#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 2 ] && [ $# -n 3]; then
  echo "Usage: $0 <data-root> <score-dir> [suffix]"
  exit 1;
fi

set -e

xvector_scp=$1
preproc_model_path=$2
utt2lang_path=$3
results_dir=$4

python3 local/did_eval.py \
        --xvector_scp $xvector_scp \
        --preproc_model_path $preproc_model_path \
        --utt2lang_path $utt2lang_path\
        --results_dir $result_dir \
    
wait

