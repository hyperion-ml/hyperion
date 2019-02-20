#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 4 ]; then
  echo "Usage: $0 <scorer-dir> <data-root> <dev/eval> <score-dir>"
  exit 1;
fi

set -e

scorer_dir=$1
data_dir=$2
dev_eval=$3
score_dir=$4

score_base=$score_dir/voices19_challenge_${dev_eval}

sed -e 's@ target$@ tgt@' -e 's@ nontarget$@ imp@' $data_dir/trials > ${score_base}_key

python2 $scorer_dir/score_voices ${score_base}_scores ${score_base}_key > ${score_base}_results

rm ${score_base}_key

