#!/bin/bash
#               2019  Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
nj=30
cmd="run.pl"
stage=0
ahc_opts="--threshold 0"
win_start=-0.625
win_length=1.5
win_shift=0.25
win_shrink=0.625
timestamps_file=""
vad_opts=""

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e 

if [ $# != 6 ];then
    echo "Usage: $0 <utt-list> <embeddings-file> <vad-file> <preproc-file> <plda-file> <output-dir>"
    echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
    echo "  --stage <stage|0>                                # To control partial reruns"
    echo "  --ahc-opts <ahc-opts|--threshold 0>              # options for clustering"
    echo "  --timestamps-file <timestamps_file|>             # scp file pointing to x-vector timestamps"
    echo "                                                     dicard win-start, win-length and win-shift"
    echo "  --win-start <win-start|-0.625>                   # xvector sliding window starting point"
    echo "  --win-length <win-lenght|1.5>                    # xvector sliding window length"
    echo "  --win-shift <win-shift|0.25>                     # xvector sliding window shift"
fi

utts=$1
v_file=$2
vad_file=$3
preproc_file=$4
plda_file=$5
output_dir=$6

log_dir=$output_dir/log
mkdir -p $log_dir

if [ -n "$timestamps_file" ];then
    ts_opts="--timestamps-file $timestamps_file"
else
    ts_opts="--win-start $win_start --win-length $win_length --win-shift $win_shift"
fi


if [ $stage -le 0 ];then
    $cmd JOB=1:$nj $output_dir/log/eval_ahc.JOB.log \
	hyp_utils/conda_env.sh \
	steps_diar/eval-ahc-v1.py \
	--test-list $utts \
	--v-file scp:$v_file \
	--vad-file $vad_file \
	--preproc-file $preproc_file \
	--model-file $plda_file \
	--rttm-file $output_dir/rttm.JOB \
	--part-idx JOB --num-parts $nj \
	$ahc_opts $ts_opts $vad_opts --win-shrink $win_shrink

fi

if [ $stage -le 1 ];then
    echo "$0: combining rttm across jobs"
    for j in $(seq $nj); do cat $output_dir/rttm.$j; done > $output_dir/rttm || exit 1;
fi

