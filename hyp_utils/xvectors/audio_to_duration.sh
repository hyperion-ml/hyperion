#!/bin/bash
#
#           2022 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.
set -e
nj=40
cmd="run.pl"
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 <data-dir>"
  echo "e.g.: $0 data/train data/train_no_sil"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data_in=$1
output_dir=$data_in/durations

name=`basename $data_in`

for f in $data_in/wav.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

mkdir -p $output_dir/log

$cmd JOB=1:$nj $output_dir/log/audio_to_duration.JOB.log \
    hyp_utils/conda_env.sh \
    audio_to_duration.py \
    --audio-file $data_in/wav.scp \
    --output-file $output_dir/utt2dur.JOB

for n in $(seq $nj); do
  cat $output_dir/utt2dur.$n || exit 1;
done > ${data_in}/utt2dur || exit 1

echo "$0: Succeeded processing audios for $name"
