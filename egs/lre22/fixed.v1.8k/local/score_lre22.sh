#!/bin/bash

. path.sh

if [ $# -ne 3 ];then
  echo "Usage: $0 <dev/eval> <score-file> <output-file>"
  exit 1
fi

dev_eval=$1
score_file=$(readlink -f $2)
output_file=$(readlink -f $3)
echo $dev_eval $score_file $output_file
output_dir=$(dirname $output_file)
mkdir -p $output_dir

conda activate $HYP_ENV

cd ./lre-scorer
echo "Scoring $score_file -> $output_file"
if [ "$dev_eval" == "dev" ];then
  config=config.ini
else
  config=config_eval.ini
fi

python ./scoreit.py  -s $score_file -o $output_file -e $config

cd -
