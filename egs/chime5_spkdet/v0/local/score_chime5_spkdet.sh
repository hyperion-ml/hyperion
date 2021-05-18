#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 2 ]; then
  echo "Usage: $0 <data-root> <score-dir>"
  exit 1;
fi

set -e

data_dir=$1
score_dir=$2

# chime5 trials
trials=$data_dir/trials

echo "Chime5 global"
python local/score_dcf.py --key-file $trials --score-file $score_dir/chime5_spkdet_scores --output-path $score_dir/chime5_spkdet &

for cond in BIN.SUM U01.CH1 U02.CH1 U04.CH1 U06.CH1
do
    echo "Chime5 $cond"
    key=${trials}_$cond

    #Compute performance
    python local/score_dcf.py --key-file $key --score-file $score_dir/chime5_spkdet_scores --output-path $score_dir/chime5_spkdet_${cond} &
done
wait



