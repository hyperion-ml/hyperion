#!/bin/bash
#               2020  Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
nj=30
cmd="run.pl"

#chunk_length=0     # The chunk size over which the embedding is extracted.
use_gpu=false
write_utt2num_frames=false  # If true writes utt2num_frames.
center=true
norm_var=false
left_context=150
right_context=150
stage=0
use_bin_vad=false
write_x_sample=false
write_x_mean=false
write_img=false
write_z_sample=false
img_frames=400


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-model> <data> <output-dir>"
  echo " e.g.: $0 exp/xvector_nnet/model.pt.tar data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --write-utt2num-frames <true|false>              # If true, write utt2num_frames file. (default:false)"
  echo "  --write-x-mean <true|false>                      # If true, write the mean of p(x|z). (default:false)"
  echo "  --write-x-sample <true|false>                    # If true, writes a sample x ~ p(x|z). (default:false)"
  echo "  --write-z-sample <true|false>                    # If true, writes a sample x ~ q(z). (default:false)"
  echo "  --write-img <true|false>                         # If true, writes pdf images of predicted features (default:false)"
  # echo "  --chunk-length <n|0>                           # If provided, applies encoder with specified chunk-length and "
  echo "                                                   # concatenates the chunks outputs before pooling"
  echo "  --center <true|false>                            # If true, normalize means in the sliding window cmvn (default:true)"
  echo "  --norm-var <true|false>                          # If true, normalize variances in the sliding window cmvn (default:false)"
  echo "  --left-context <int>                             # Left context for short-time cmvn (default: 150)"
  echo "  --right-context <int>                            # Right context for short-time cmvn (default: 150)"
  echo "  --use-bin-vad <true|false>                       # Remove silence frames usin binary vad in vad.scp (default:false)"
  echo "  --img-frames <n|400>                             # number of frames to plot on pdf img"
  

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
    args="${args} --mvn.no-norm-mean"
fi
if [ "$norm_var" == "true" ];then
    args="${args} --mvn.norm-var"
fi

if [ "$use_bin_vad" == "true" ];then
    args="${args} --vad scp:$data_dir/vad.scp"
fi

if [ "$write_utt2num_frames" == "true" ];then
    args="${args} --write-num-frames $output_dir/utt2num_frames.JOB"
fi

if [ "$write_x_mean" == "true" ];then
    output_xmean_dir=$output_dir/xmean
    mkdir -p $output_xmean_dir
    args="${args} --write-x-mean ark,scp:$output_xmean_dir/xmean.JOB.ark,$output_xmean_dir/xmean.JOB.scp"
fi


if [ "$write_x_sample" == "true" ];then
    output_xsample_dir=$output_dir/xsample
    mkdir -p $output_xsample_dir
    args="${args} --write-x-sample ark,scp:$output_xsample_dir/xsample.JOB.ark,$output_xsample_dir/xsample.JOB.scp"
fi

if [ "$write_z_sample" == "true" ];then
    output_zsample_dir=$output_dir/zsample
    mkdir -p $output_zsample_dir
    args="${args} --write-z-sample ark,scp:$output_zsample_dir/zsample.JOB.ark,$output_zsample_dir/zsample.JOB.scp"
fi


if [ "$write_img" == "true" ];then
    output_img_dir=$output_dir/img
    mkdir -p $output_img_dir
    args="${args} --write-img-path $output_img_dir --img-frames $img_frames"
fi


if [ $stage -le 0 ];then
    $cmd JOB=1:$nj $output_dir/log/eval_vae.JOB.log \
	hyp_utils/conda_env.sh --num-gpus $num_gpus \
	torch-eval-vae.py ${args}  \
	--mvn.left-context $left_context --mvn.right-context $right_context \
	--part-idx JOB --num-parts $nj \
	--input scp:$data_dir/feats.scp \
	--model-path $nnet_file \
	--scores $output_dir/scores.JOB.csv || exit 1;
fi


if [ $stage -le 1 ]; then
  echo "$0: combining scores across jobs"
  cat $output_dir/scores.1.csv > $output_dir/scores.csv
  for j in $(seq 2 $nj);
  do
      tail -n +2 $output_dir/scores.$j.csv
  done >> $output_dir/scores.csv
  python -c "import pandas as pd; x = pd.read_csv('$output_dir/scores.csv'); x_mean = x.mean(axis=0); x_std=x.std(axis=0); x_stats=pd.DataFrame([x_mean, x_std], index=['mean','std']); x_stats.to_csv('$output_dir/scores_stats.csv'); print(x_stats)"

  for d in xmean xsample zsample
  do
      if [ -f $output_dir/$d/$d.1.scp ];then
	  echo "$0: combining $d across jobs"
	  for j in $(seq $nj); do cat $output_dir/$d/$d.$j.scp; done > $output_dir/$d/$d.scp || exit 1;
      fi
  done
  if [ "$write_utt2num_frames" == "true" ];then
      echo "$0: combining utt2num_frames across jobs"
      for n in $(seq $nj); do
	  cat $output_dir/utt2num_frames.$n || exit 1;
      done > $output_dir/utt2num_frames || exit 1
  fi
fi

