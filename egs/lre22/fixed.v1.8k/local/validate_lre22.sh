#!/bin/bash

. path.sh

if [ $# -ne 1 ];then
  echo "Usage: $0 <score-file>"
  exit 1
fi

score_file=$(readlink -f $1)
conda activate $HYP_ENV

cd ./lre-scorer
echo "Scoring $score_file -> $output_file"
python ./scoreit.py -s $score_file -o $score_file.val -v

cd -
