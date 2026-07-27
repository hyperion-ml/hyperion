#!/bin/bash
#               2019  Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
nj=30
cmd="run.pl"

chunk_length=0     # The chunk size over which the embedding is extracted.
use_gpu=false
write_utt2num_frames=true  # If true writes utt2num_frames.
center=true
norm_var=false
left_context=150
right_context=150
stage=0
min_utt_length=500
max_utt_length=12000
random_utt_length=false


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <xvec-nnet-model> <vae-nnet-model> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet/model.pt.tar data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --write-utt2num-frames <true|false>              # If true, write utt2num_frames file."
  echo "  --chunk-length <n|0>                             # If provided, applies encoder with specified chunk-length and "
  echo "                                                   # concatenates the chunks outputs before pooling"
  echo "  --center <true|false>                            # If true, normalize means in the sliding window cmvn (default:true)"
  echo "  --norm-var <true|false>                          # If true, normalize variances in the sliding window cmvn (default:false)"
  echo "  --left-context <int>                             # Left context for short-time cmvn (default: 150)"
  echo "  --right-context <int>                            # Right context for short-time cmvn (default: 150)"
  echo "  --random-utt-length                              # If true, extracts a random chunk from the utterance between min_utt_length and max_utt_length"
  echo "  --min-utt-length <n|0>                           # "
  echo "  --max-utt-length <n|0>                           # "
  

fi

xvec_nnet_file=$1
vae_nnet_file=$2
data_dir=$3
output_dir=$4

for f in $data_dir/feats.scp $data_dir/vad.scp ; do
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
    args="${args} --mvn.no-norm-mean"
fi
if [ "$norm_var" == "true" ];then
    args="${args} --mvn.norm-var"
fi

if [ "$random_utt_length" == "true" ];then
    args="${args} --random-utt-length --min-utt-length $min_utt_length --max-utt-length $max_utt_length"
fi

if [ "$write_utt2num_frames" == "true" ];then
    write_num_frames_opt="--write-num-frames $output_dir/utt2num_frames.JOB"
fi

if [ $stage -le 0 ];then
    $cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	hyp_utils/conda_env.sh --num-gpus $num_gpus \
	torch-extract-xvectors-vae-preproc.py ${args} $write_num_frames_opt \
	--mvn.left-context $left_context --mvn.right-context $right_context \
	--part-idx JOB --num-parts $nj \
	--input scp:$data_dir/feats.scp --vad scp:$data_dir/vad.scp \
	--xvec-model-path $xvec_nnet_file --vae-model-path $vae_nnet_file --chunk-length $chunk_length \
	--output ark,scp:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.scp || exit 1;
fi


if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $output_dir/xvector.$j.scp; done > $output_dir/xvector.scp || exit 1;
  if [ "$write_utt2num_frames" == "true" ];then
      for n in $(seq $nj); do
	  cat $output_dir/utt2num_frames.$n || exit 1;
      done > $output_dir/utt2num_frames || exit 1
  fi
fi

