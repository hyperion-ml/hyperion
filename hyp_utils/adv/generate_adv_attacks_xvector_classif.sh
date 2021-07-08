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
min_utt_length=500
max_utt_length=12000
random_utt_length=false
stage=1

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 5 ]; then
  echo "Usage: $0 [options] <nnet-file> <data-dir> <class2int-file> <attack-tag> <output-dir>"
  echo "Options: "
  echo "  --feat-config <config-file>                      # feature extractor config"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --attack-opts <str>                               # other options for the attack"
  echo "  --use-bin-vad <bool|true>                        # If true, uses binary VAD from vad.scp"
  echo "  --random-utt-length                              # If true, extracts a random chunk from the utterance between "
  echo "                                                   # min_utt_length and max_utt_length"
  echo "  --min-utt-length <n|0>                           # "
  echo "  --max-utt-length <n|0>                           # "
  exit 1;
fi

nnet_file=$1
data_dir=$2
class2int=$3
attack_tag=$4
output_dir=$5

log_dir=$output_dir/log

mkdir -p $log_dir
wav=$data_dir/wav.scp
vad=$data_dir/vad.scp
list=$data_dir/utt2spk

required="$wav $list"

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
if [ "$random_utt_length" == "true" ];then
    args="${args} --random-utt-length --min-utt-length $min_utt_length --max-utt-length $max_utt_length"
fi

echo "$0: generate attacks for $data_dir to $output_dir"

if [ $stage -le 1 ];then
    $cmd JOB=1:$nj $log_dir/generate_attack.JOB.log \
	hyp_utils/conda_env.sh --num-gpus $num_gpus \
	torch-generate-adv-attacks-xvector-classif.py \
	--feats $feat_config ${args} $attacks_opts \
	--wav-file $wav \
	--list-file $list \
	--model-path $nnet_file \
	--class2int-file $class2int \
	--attack-tag $attack_tag \
	--output-wav-dir $output_dir/wav \
	--attack-info-file $output_dir/info/info.JOB.yaml \
	--part-idx JOB --num-parts $nj || exit 1
fi

if [ $stage -le 2 ];then
    for((j=1;j<=$nj;j++));
    do
	cat $output_dir/info/info.$j.yaml
    done > $output_dir/info/info.yaml
fi


