#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
nj=40
fps=1
use_gpu=false
det_window=21
time_in_secs=false
thr_d=0

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data-dir> <facedet-model-dir> <faceembed-model-dir> <output-dir>"
  exit 1;
fi

data_dir=$1
facedet_model=$2
faceembed_model=$3
output_dir=$4

logdir=$output_dir/log
mkdir -p $logdir

name=`basename $data_dir`
scp=$data_dir/vid.scp
bbox_file=$data_dir/face_boundingbox

num_gpus=0
if [ "$use_gpu" == "true" ];then
  num_gpus=1
  args="--use-gpu"
  cmd="$cmd --gpu 1"
fi

if [ "$time_in_secs" == "true" ];then
    args=$args" --time-in-secs"
fi

echo "Extracting face embeddings $data_dir to $output_dir"

$cmd JOB=1:$nj $logdir/face_embed_${name}.JOB.log \
     hyp_utils/conda_env.sh --conda-env $MXNET_ENV --num-gpus $num_gpus \
     steps_insightface/extract-face-embed-from-bbox-plus-face-det-v4.py \
     --input-path $scp \
     --output-path $output_dir/embed.JOB \
     --bbox-path $bbox_file \
     --facedet-model-file $facedet_model \
     --faceembed-model-file $faceembed_model \
     --fps $fps --save-facedet-img --save-facecrop-img \
     --det-window $det_window --thr-d $thr_d $args \
     --part-idx JOB --num-parts $nj || exit 1


for((i=1;i<=$nj;i++))
do
  cat $output_dir/embed.$i.scp
done > $output_dir/embed.scp

for((i=1;i<=$nj;i++))
do
  cat $output_dir/embed.$i.bbox
done > $output_dir/embed.bbox



