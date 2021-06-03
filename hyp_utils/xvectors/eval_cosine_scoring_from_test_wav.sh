#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
nj=20
cmd=run.pl
feat_config=conf/fbank80_stmn_16k.yaml
use_gpu=false
cal_file=""
max_test_length=""

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 6 ]; then
  echo "Usage: $0 [options] <ndx> <enroll-file> <test-data-dir> <vector-file> <nnet-model> <output-scores>"
  echo "Options: "
  echo "  --feat-config <config-file>                      # feature extractor config"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --cal-file <str|>                                # calibration params file"
  echo "  --max-test-length <float|>                       # maximum length for the test side in secs"
  exit 1;
fi

ndx_file=$1
enroll_file=$2
test_data=$3
vector_file=$4
nnet_file=$5
output_file=$6

output_dir=$(dirname $output_file)
log_dir=$output_dir/log

mkdir -p $log_dir
name=$(basename $output_file)

wav=$test_data/wav.scp
vad=$test_data/vad.scp

required="$wav $feat_config $ndx $enroll_file $vector_file $vad"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

num_gpus=0
args=""
if [ "$use_gpu" == "true" ];then
    cmd="$cmd --gpu 1"
    num_gpus=1
    args="--use-gpu"
fi

if [ -n "$cal_file" ];then
    args="${args} --cal-file $cal_file"
fi

if [ -n "$max_test_length" ];then
    args="${args} --max-test-length $max_test_length"
fi

echo "$0: score $ndx_file to $output_dir"

$cmd JOB=1:$nj $log_dir/${name}.JOB.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $num_gpus \
    torch-eval-xvec-cosine-scoring-from-test-wav.py \
    --feats $feat_config ${args} \
    --v-file scp:$vector_file \
    --ndx-file $ndx_file \
    --enroll-file $enroll_file \
    --test-wav-file $wav \
    --vad scp:$vad \
    --model-path $nnet_file \
    --score-file $output_file \
    --seg-part-idx JOB --num-seg-parts $nj || exit 1


for((j=1;j<=$nj;j++));
do
    cat $output_file-$(printf "%03d" 1)-$(printf "%03d" $j)
done | sort -u > $output_file



