#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
nj=40
fps=1
use_gpu=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-dir> <model-dir> <output-dir>"
  exit 1;
fi

data_dir=$1
model=$2
output_dir=$3

logdir=$output_dir/log
mkdir -p $logdir

name=`basename $data_dir`
scp=$data_dir/vid.scp

num_gpus=0
if [ "$use_gpu" == "true" ];then
  num_gpus=1
  args="--use-gpu"
  cmd="$cmd --gpu 1"
fi

$cmd JOB=1:$nj $logdir/face_det_${name}.JOB.log \
     hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $num_gpus \
     steps_insightface/eval-face-det-v1.py $args \
     --input-path $scp --output-path $output_dir/$name.JOB \
     --model-file $model --fps $fps --save-face-img \
     --part-idx JOB --num-parts $nj || exit 1


