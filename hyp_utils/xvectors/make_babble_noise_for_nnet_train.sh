#!/bin/bash
#
#           2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.
set -e

nj=1
cmd="run.pl"
stage=0
file_format=flac
nodes=b1
storage_name=$(date +'%m_%d_%H_%M')
#proc_opts="--remove-dc-offset"
min_spks=3
max_spks=10
num_reuses=5

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir> <feat-dir>"
  echo "e.g.: $0 data/train data/train_no_sil exp/make_xvector_features"
  echo "Options: "
  #echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --file-format <str|flac>                         # Output file_format supported by soundfile (flac,ogg,wav,...)"
  #echo "  --proc-opts <str|--remove-dc-offset>             # Extra arguments for proc-audio-files.py"
  echo "  --min-spks <int|3>                               # max number of spks per utterance"
  echo "  --max-spks <int|10>                              # max number of spks per utterance"
  echo "  --num-reuses <int|10>                            # number of times a signal is reused to create babble"
  exit 1;
fi

data_in=$1
data_out=$2
dir=$3

name=`basename $data_in`

for f in $data_in/wav.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $data_out
output_dir=$(utils/make_absolute.sh $dir)

args=""
$cmd $dir/log/make_babble_noise_${name}.log \
    hyp_utils/conda_env.sh \
    make_babble_noise_audio_files.py ${args} \
    --output-audio-format $file_format $args $proc_opts \
    --min-spks $min_spks --max-spks $max_spks --num-reuses $num_reuses \
    --write-time-durs $data_out/utt2dur \
    --input $data_in/wav.scp \
    --output-path $output_dir \
    --output-script $data_out/wav.scp



# for n in $(seq $nj); do
#   cat $output_dir/wav.${name}.$n.scp || exit 1;
# done > ${data_out}/wav.scp || exit 1

# for n in $(seq $nj); do
#   cat $output_dir/utt2dur.${name}.$n || exit 1;
# done > ${data_out}/utt2dur || exit 1

echo "$0: Succeeded making babble noise for $name"
