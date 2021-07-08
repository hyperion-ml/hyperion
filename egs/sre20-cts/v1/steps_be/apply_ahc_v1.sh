#!/bin/bash
# Copyright 2018 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

cmd=run.pl
plda_type=frplda
ncoh=0
thr=0
cal_file=""
min_sps=4
class_prefix=""
set -e

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 5 ]; then
  echo "Usage: $0 <data-dir> <vector-file> <preproc-file> <plda-file> <output-data-dir>"
  exit 1;
fi

data_dir=$1
vector_file=$2
preproc_file=$3
plda_file=$4
output_dir=$5

rm -rf $output_dir

mkdir -p $output_dir/log
name=$(basename $output_dir)

for f in utt2spk; do
  if [ ! -f $data_dir/$f ]; then
    echo "$0: no such file $data_dir/$f"
    exit 1;
  fi
done

train_list=$output_dir/train_utt2spk
awk -v fv=$vector_file 'BEGIN{
while(getline < fv)
{
   files[$1]=1
}
}
{ if ($1 in files) {print $1,$2}}' $data_dir/utt2spk > $train_list


echo "$0 clustering $data_dir"

extra_args=""
if [ -n "$cal_file" ];then
    extra_args="--cal-file $cal_file"
fi

if [ -n "$class_prefix" ];then
    extra_args="${extra_args} --class-prefix $class_prefix"
fi


$cmd $output_dir/log/ahc.log \
     hyp_utils/conda_env.sh steps_be/apply-ahc-v1.py \
     --v-file scp:$vector_file \
     --input-list $train_list \
     --preproc-file $preproc_file \
     --model-file $plda_file \
     --plda-type $plda_type \
     --coh-nbest $ncoh \
     --threshold $thr \
     --score-hist-file $output_dir/score_hist.pdf \
     --output-list $output_dir/utt2spk $extra_args

rm $train_list
utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt
for f in utt2lang vad.scp wav.scp utt2num_frames utt2dur
do
    if [ -f $data_dir/$f ];then
	cp $data_dir/$f  $output_dir/$f
    fi
done
#awk -v min_sps=$min_sps 'NF > min_sps { print $0 }'  $output_dir/spk2utt > $output_dir/spk2utt_minsps$min_sps
#utils/spk2utt_to_utt2spk.pl $output_dir/spk2utt_minsps$min_sps > $output_dir/utt2spk_minsps$min_sps
