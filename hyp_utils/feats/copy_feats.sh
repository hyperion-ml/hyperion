#!/bin/bash
#
# Copyright 2017 Johns Hopkins University (David Snyder)
#           2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0.

# This script applies sliding window cmvn and removes silence frames.  This
# is performed on the raw features prior to generating examples for training
# the xvector system.

nj=40
cmd="run.pl"
stage=0
compress=true
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
  exit 1;
fi

data_in=$1
data_out=$2
dir=$3

name=`basename $data_in`

for f in $data_in/feats.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $data_out
featdir=$(utils/make_absolute.sh $dir)

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $featdir/storage ]; then
    dir_name=$USER/hyp-data/kaldi-xvector/$storage_name/xvector_feats/storage
    if [ "$nodes" == "b0" ];then
	utils/create_split_dir.pl \
	    utils/create_split_dir.pl \
	    /export/b{04,05,06,07}/$dir_name $featdir/storage
    elif [ "$nodes" == "b1" ];then
	utils/create_split_dir.pl \
	    /export/b{14,15,16,17,18}/$dir_name $featdir/storage
    else
	utils/create_split_dir.pl \
	    /export/c{06,07,08,09}/$dir_name $featdir/storage
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

opt_args=""
if [ "$compress" == "true" ];then
    opt_args="${opt_args} --compress"
fi
write_num_frames_opt="--write-num-frames $featdir/log/utt2num_frames.JOB"

$cmd JOB=1:$nj $dir/log/copy_feats_${name}.JOB.log \
     copy-feats.py ${opt_args} $write_num_frames_opt \
     --part-idx JOB --num-parts $nj \
     --input scp:$data_in/feats.scp \
     --output h5,scp:$featdir/feats_${name}.JOB.h5,$featdir/feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1

for n in $(seq $nj); do
  cat $featdir/log/utt2num_frames.$n || exit 1;
done > $data_out/utt2num_frames || exit 1
rm $featdir/log/utt2num_frames.*

echo "$0: Succeeded creating features for $name"
