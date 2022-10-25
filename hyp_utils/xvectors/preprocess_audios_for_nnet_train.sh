#!/bin/bash
#
#           2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.
set -e
nj=40
cmd="run.pl"
stage=0
file_format=flac
nodes=b1
storage_name=$(date +'%m_%d_%H_%M')
proc_opts="--remove-dc-offset"
use_bin_vad=false

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir> <feat-dir>"
  echo "e.g.: $0 data/train data/train_no_sil exp/make_xvector_features"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --file-format <str|flac>                         # Output format supported by soundfile (flac,ogg,wav,...)"
  echo "  --proc-opts <str|--remove-dc-offset>             # Extra arguments for proc-audio-files.py"
  echo "  --use-bin-vad <true,false|false>                 # Removes silence using binary vad"
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

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $output_dir/storage ]; then
    dir_name=$USER/hyp-data/$storage_name/xvector_audio/storage
    if [ "$nodes" == "b0" ];then
	utils/create_split_dir.pl \
	    utils/create_split_dir.pl \
	    /export/b{04,05,06,07}/$dir_name $output_dir/storage
    elif [ "$nodes" == "b1" ];then
	utils/create_split_dir.pl \
	    /export/b{14,15,16,17,18}/$dir_name $output_dir/storage
    elif [ "$nodes" == "s01" ];then
	utils/create_split_dir.pl \
	    /export/s01/$dir_name $output_dir/storage
    else
	utils/create_split_dir.pl \
	    /export/c{01,06,07,08,09}/$dir_name $output_dir/storage
    fi

    for f in $(awk '{ print $1}' $data_in/wav.scp); do
	# the next command does nothing unless $output_dir/storage/ exists, see
	# utils/create_data_link.pl for more info.
	utils/create_data_link.pl $output_dir/$f.$file_format
    done
fi


for f in utt2spk spk2utt wav.scp utt2lang spk2gender
do
    if [ -f $data_in/$f ];then
	cp $data_in/$f $data_out/$f
    fi
done

args=""
if [ "$use_bin_vad" == "true" ];then
    args="${args} --vad scp:$data_in/vad.scp"
else
    f=vad.scp
    if [ -f $data_in/$f ];then
	cp $data_in/$f $data_out/$f
    fi
fi

$cmd JOB=1:$nj $dir/log/preproc_audios_${name}.JOB.log \
    hyp_utils/conda_env.sh \
    preprocess_audio_files.py ${args} --output-audio-format $file_format $args $proc_opts \
    --write-time-durs $output_dir/utt2dur.${name}.JOB \
    --part-idx JOB --num-parts $nj \
    --input $data_in/wav.scp \
    --output-path $output_dir \
    --output-script $output_dir/wav.${name}.JOB.scp

for n in $(seq $nj); do
  cat $output_dir/wav.${name}.$n.scp || exit 1;
done > ${data_out}/wav.scp || exit 1

for n in $(seq $nj); do
  cat $output_dir/utt2dur.${name}.$n || exit 1;
done > ${data_out}/utt2dur || exit 1

echo "$0: Succeeded processing audios for $name"
