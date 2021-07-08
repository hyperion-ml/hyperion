#!/bin/bash
#
# Copyright 2019 Johns Hopkins University (Jesus Villalba)
#           
# Apache 2.0.

# This script applies sliding window cmvn and removes silence frames.  This
# is performed on the raw features prior to generating examples for training
# the xvector system.
set -e 

nj=40
cmd="run.pl"
stage=0
center=true
norm_var=false
compression_method=auto
file_format=h5
compress=true
left_context=150
right_context=150
nodes=b1
storage_name=$(date +'%m_%d_%H_%M')

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir> <feat-dir>"
  echo "e.g.: $0 data/train data/train_no_sil exp/make_xvector_features"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --norm-mean <true|false>                         # If true, normalize means in the sliding window cmvn (default:true)"
  echo "  --norm-var <true|false>                          # If true, normalize variances in the sliding window cmvn (default:false)"
  echo "  --left-context <int>                             # Left context for short-time cmvn (default: 150)"
  echo "  --right-context <int>                            # Right context for short-time cmvn (default: 150)"
  exit 1;
fi

data_in=$1
data_out=$2
dir=$3

name=`basename $data_in`

for f in $data_in/feats.scp $data_in/vad.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $data_out
featdir=$(utils/make_absolute.sh $dir)

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $featdir/storage ]; then
    dir_name=$USER/hyp-data/xvectors/$storage_name/xvector_feats/storage
    if [ "$nodes" == "b0" ];then
	utils/create_split_dir.pl \
	    utils/create_split_dir.pl \
	    /export/b{04,05,06,07}/$dir_name $featdir/storage
    elif [ "$nodes" == "b1" ];then
	utils/create_split_dir.pl \
	    /export/b{14,15,16,17,18}/$dir_name $featdir/storage
    else
	utils/create_split_dir.pl \
	    /export/c{01,06,07,08,09}/$dir_name $featdir/storage
    fi
fi

for n in $(seq $nj); do
  # the next command does nothing unless $featdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $featdir/feats_${name}.${n}.h5
done

for f in utt2spk spk2utt wav.scp utt2lang spk2gender
do
    if [ -f $data_in/$f ];then
	cp $data_in/$f $data_out/$f
    fi
done

args=""
if [ "$center" == "false" ];then
    args="${args} --no-norm-mean"
fi
if [ "$norm_var" == "true" ];then
    args="${args} --norm-var"
fi
if [ "$compress" == "true" ];then
    args="${args} --compress --compression-method $compression_method"
fi
write_num_frames_opt="--write-num-frames $featdir/log/utt2num_frames.JOB"

$cmd JOB=1:$nj $dir/log/create_embed_feats_${name}.JOB.log \
    hyp_utils/conda_env.sh apply-mvn-select-frames.py ${args} $write_num_frames_opt \
     --left-context $left_context --right-context $right_context \
     --part-idx JOB --num-parts $nj \
     --input scp:$data_in/feats.scp --vad scp:$data_in/vad.scp \
     --output ${file_format},scp:$featdir/feats_${name}.JOB.${file_format},$featdir/feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1

for n in $(seq $nj); do
  cat $featdir/log/utt2num_frames.$n || exit 1;
done > $data_out/utt2num_frames || exit 1
rm $featdir/log/utt2num_frames.*

echo "$0: Succeeded creating features for $name"
