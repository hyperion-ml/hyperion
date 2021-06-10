#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
nj=20
cmd=run.pl
feat_config=conf/fbank80_stmn_16k.yaml
use_gpu=false
attacks_opts=""
use_bin_vad=true
cal_file=""
threshold=0
stage=1

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 7 ]; then
  echo "Usage: $0 [options] <nnet-file> <key> <enroll-file> <test-data-dir> <vector-file> <attack-tag> <output-dir>"
  echo "Options: "
  echo "  --feat-config <config-file>                      # feature extractor config"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --attack-opts <str>                               # other options for the attack"
  echo "  --use-bin-vad <bool|true>                        # If true, uses binary VAD from vad.scp"
  echo "  --threshold <float|0>                            # decision threshold"
  echo "  --cal-file <str|>                                # calibration params file"
  exit 1;
fi

nnet_file=$1
key_file=$2
enroll_file=$3
data_dir=$4
vector_file=$5
attack_tag=$6
output_dir=$7

log_dir=$output_dir/log

mkdir -p $log_dir
wav=$data_dir/wav.scp
vad=$data_dir/vad.scp

required="$wav"

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

if [ "$use_bin_vad" == "true" ];then
    args="${args} --vad scp:$vad"
fi
if [ -n "$cal_file" ];then
    args="${args} --cal-file $cal_file"
fi

echo "$0: generate attacks for $data_dir to $output_dir"

if [ $stage -le 1 ];then
    $cmd JOB=1:$nj $log_dir/generate_attack.JOB.log \
	hyp_utils/conda_env.sh --num-gpus $num_gpus \
	torch-generate-adv-attacks-xvector-verif.py \
	--feats $feat_config ${args} $attacks_opts \
	--v-file scp:$vector_file \
	--key-file $key_file \
	--enroll-file $enroll_file \
	--test-wav-file $wav \
	--model-path $nnet_file \
	--attack-tag $attack_tag \
	--output-wav-dir $output_dir/wav \
	--attack-info-file $output_dir/info/info.JOB.yaml \
	--threshold $threshold \
	--seg-part-idx JOB --num-seg-parts $nj || exit 1
fi

if [ $stage -le 2 ];then
    for((j=1;j<=$nj;j++));
    do
	cat $output_dir/info/info.$j.yaml
    done > $output_dir/info/info.yaml
fi


