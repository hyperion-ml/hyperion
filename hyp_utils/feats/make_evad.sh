#!/bin/bash
# Copyright    2019       Johns Hopkins University (Author: Jesus Villalba)
#              2012-2016  Karel Vesely  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration section.
nj=4
cmd=run.pl
vad_config=conf/vad.yml
write_utt2num_frames=false  # if true writes utt2num_frames
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<vad-dir>] ]";
   echo "e.g.: $0 data/train exp/make_vad/train vad"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <vad-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --vad-config <config-file>                     # config passed to compute-vad-feats "
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
  vaddir=$3
else
  vaddir=$data/data
fi

# make $vaddir an absolute pathname.
vaddir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $vaddir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $vaddir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $vad_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_vad.sh: no such file $f"
    exit 1;
  fi
done

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $vaddir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $vaddir/vad_$name.$n.ark
done

opt_args=""

if $write_utt2num_frames; then
  opt_args="${opt_args} --write-num-frames $data/utt2num_frames.JOB"
fi

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  opt_args="${opt_args} --segments $data/segments"
fi

set +e
$cmd JOB=1:$nj $logdir/make_vad_${name}.JOB.log \
    hyp_utils/conda_env.sh \
    compute_energy_vad.py --cfg $vad_config $opt_args \
    --input $scp --output ark,scp:$vaddir/vad_$name.JOB.ark,$vaddir/vad_$name.JOB.scp \
    --part-idx JOB --num-parts $nj
set -e

# rerun not successful jobs
for tmp in {1..3};do
    pids=""

    for((i=1;i<=$nj;i++))
    do
	status=$(tail -n 1 $logdir/make_vad_${name}.$i.log | \
			awk '/status 0/ { print 0}
                            !/status 0/ { print 1}')
	if [ $status -eq 1 ];then
	    echo "JOB $i failed, resubmitting"
        sleep 10
        opt_args=`echo ${opt_args} | sed -e "s/utt2num_frames.JOB/utt2num_frames.$i/g"`
        $cmd $logdir/make_vad_${name}.$i.log \
            hyp_utils/conda_env.sh \
            compute_energy_vad.py --cfg $vad_config $opt_args \
            --input $scp --output ark,scp:$vaddir/vad_$name.$i.ark,$vaddir/vad_$name.$i.scp \
            --part-idx $i --num-parts $nj &
        opt_args=`echo ${opt_args} | sed -e "s/utt2num_frames.$i/utt2num_frames.JOB/g"`
        pids="$pids $!"
	fi
    done

    for pid in $pids;do
        wait $pid
    done
done
wait

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $vaddir/vad_$name.$n.scp || exit 1;
done > $data/vad.scp

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $data/utt2num_frames.$n || exit 1;
  done > $data/utt2num_frames || exit 1
  rm $data/utt2num_frames.*
fi

nf=`cat $data/vad.scp | wc -l`
nu=`cat $scp | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating binary VAD for $name"
