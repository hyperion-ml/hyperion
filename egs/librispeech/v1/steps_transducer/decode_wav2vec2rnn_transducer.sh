#!/bin/bash
#               2022  Johns Hopkins University (Author: Yen-Ju Lu)
# Apache 2.0.
nj=30
cmd="run.pl"
set -e
use_gpu=false
#write_utt2num_frames=true  # If true writes utt2num_frames.
stage=0
extra_args=""
infer_cfg=conf/infer.yaml
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ] && [ $# != 4 ]; then
  echo "Usage: $0 [options] <nnet-model> <data> <xvector-dir> [<data-out-dir>]"
  echo " e.g.: $0 --feat-config conf/fbank_mvn.yml --aug-config conf/noise_aug.yml exp/xvector_nnet/model.pt data/train exp/xvectors_train [data/train_aug]"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --infer-cfg <file|conf/infer.yaml>               # decoding configuration"
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  

fi

nnet_file=$1
data_dir=$2
output_dir=$3
bpe_model=$4

for f in $data_dir/wav.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

log_dir=$output_dir/log
mkdir -p $log_dir

num_gpus=0
if [ "$use_gpu" == "true" ];then
    cmd="$cmd --gpu 1"
    num_gpus=1
    extra_args="${extra_args} --use-gpu"
fi 

# if [ "$write_utt2num_frames" == "true" ];then
#     write_num_frames_opt="--write-num-frames $output_dir/utt2num_frames.JOB"
# fi

if [ $stage -le 0 ];then
    $cmd JOB=1:$nj $output_dir/log/decode_transducer.JOB.log \
	hyp_utils/conda_env.sh --num-gpus $num_gpus \
	decode_wav2vec2rnn_transducer.py \
	--infer-args $infer_cfg \
	--part-idx JOB --num-parts $nj \
	--input $data_dir/wav.scp \
	--model-path $nnet_file \
	--bpe-model $bpe_model \
	--output $output_dir/transducer.JOB.text $extra_args
fi

if [ $stage -le 1 ];then
  echo "compute wer"
  cat $output_dir/transducer.*.text > $output_dir/transducer.text
  compute-wer --text --mode=present ark:$data_dir/text ark:$output_dir/transducer.text
fi
