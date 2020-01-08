#!/bin/bash
# Copyright    2019       Johns Hopkins University (Author: Jesus Villalba)
#              2012-2016  Karel Vesely  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration section.
nj=4
cmd=run.pl
mfcc_config=conf/mfcc.conf
compress=true
write_utt2num_frames=false  # if true writes utt2num_frames
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<mfcc-dir>] ]";
   echo "e.g.: $0 data/train exp/make_mfcc/train mfcc"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <mfcc-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --mfcc-config <config-file>                     # config passed to compute-mfcc-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --write-utt2num-frames <true|false>     # If true, write utt2num_frames file."
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  mfccdir=$3
else
  mfccdir=$data/data
fi


# make $mfccdir an absolute pathname.
mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mfccdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $mfcc_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mfcc.sh: no such file $f"
    exit 1;
  fi
done

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $mfccdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $mfccdir/raw_mfcc_$name.$n.ark
done

opt_args=""

if $write_utt2num_frames; then
  opt_args="${opt_args} --write-num-frames $data/utt2num_frames.JOB"
fi

if $compress;then
  opt_args="${opt_args} --compress"
fi


if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  opt_args="${opt_args} --segments $data/segments"
fi

$cmd JOB=1:$nj $logdir/make_mfcc_${name}.JOB.log \
    compute-mfcc-feats.py @$mfcc_config $opt_args \
    --input $scp --output ark,scp:$mfccdir/raw_mfcc_$name.JOB.ark,$mfccdir/raw_mfcc_$name.JOB.scp \
    --part-idx JOB --num-parts $nj || exit 1


# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $mfccdir/raw_mfcc_$name.$n.scp || exit 1;
done > $data/feats.scp

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $data/utt2num_frames.$n || exit 1;
  done > $data/utt2num_frames || exit 1
  rm $data/utt2num_frames.*
fi

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating MFCC features for $name"
