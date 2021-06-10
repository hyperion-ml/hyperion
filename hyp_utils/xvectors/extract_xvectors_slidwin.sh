#!/bin/bash

#               2019  Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
nj=30
cmd="run.pl"

chunk_length=0     # The chunk size over which the embedding is extracted.
use_gpu=false
write_timestamps=false  # If true writes xvector time-stamps
center=true
norm_var=false
left_context=150
right_context=150
stage=0
win_length=1.5
win_shift=0.25
#fs=16000
snip_edges=false
feat_opts="--feat-frame-length 25 --feat-frame-shift 10"
use_bin_vad=false

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-model> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet/model.pt.tar data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  #echo "  --write-utt2num-frames <true|false>              # If true, write utt2num_frames file."
  echo "  --chunk-length <n|0>                             # If provided, applies encoder with specified chunk-length and "
  echo "                                                   # concatenates the chunks outputs before pooling"
  echo "  --center <true|false>                            # If true, normalize means in the sliding window cmvn (default:true)"
  echo "  --norm-var <true|false>                          # If true, normalize variances in the sliding window cmvn (default:false)"
  echo "  --left-context <int>                             # Left context for short-time cmvn (default: 150)"
  echo "  --right-context <int>                            # Right context for short-time cmvn (default: 150)"
  #echo "  --random-utt-length                              # If true, extracts a random chunk from the utterance between min_utt_length and max_utt_length"
  #echo "  --min-utt-length <n|0>                           # "
  #echo "  --max-utt-length <n|0>                           # "
  

fi

nnet_file=$1
data_dir=$2
output_dir=$3

for f in $data_dir/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

log_dir=$output_dir/log
mkdir -p $log_dir

num_gpus=0
args=""
if [ "$use_gpu" == "true" ];then
    cmd="$cmd --gpu 1"
    num_gpus=1
    args="--use-gpu"
fi

if [ "$center" == "false" ];then
    args="${args} --no-norm-mean"
fi
if [ "$norm_var" == "true" ];then
    args="${args} --norm-var"
fi

if [ "$snip_edges" == "true" ];then
    args="${args} --snip-edges"
fi

if [ "$use_bin_vad" == "true" ];then
    args="${args} --vad scp:$data_dir/vad.scp"
fi

if [ "$write_timestamps" == "true" ];then
    write_timestamps_opt="--write-timestamps ark,scp:$output_dir/timestamps.JOB.ark,$output_dir/timestamps.JOB.scp"
fi

if [ $stage -le 0 ];then
    $cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	hyp_utils/conda_env.sh --num-gpus $num_gpus \
	torch-extract-xvectors-slidwin.py ${args} $write_timestamps_opt \
	--left-context $left_context --right-context $right_context \
	--part-idx JOB --num-parts $nj \
	--input scp:$data_dir/feats.scp \
	--model-path $nnet_file --chunk-length $chunk_length \
	--win-length $win_length --win-shift $win_shift $feat_opts \
	--slidwin-params-path $output_dir/slidwin.JOB.yml \
	--output ark,scp:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.scp || exit 1;
fi


if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $output_dir/xvector.$j.scp; done > $output_dir/xvector.scp || exit 1;
  if [ "$write_timestaps" == "true" ];then
      for j in $(seq $nj); do cat $output_dir/timestamps.$j.scp; done > $output_dir/timestamps.scp || exit 1;
  fi
  cp $output_dir/slidwin.1.yml $output_dir/slidwin.yml
fi

