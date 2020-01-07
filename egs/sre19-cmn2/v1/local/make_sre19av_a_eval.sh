#!/bin/bash

# Copyright 2018 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SRE19_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
enroll_file=$docs/sre19_av_eval_enrollment.tsv
enroll_diar_file=$docs/sre19_av_eval_enrollment_diarization.tsv
segm_file=$docs/sre19_av_eval_segment_key.tsv
trial_file=$docs/sre19_av_eval_trials.tsv
key_file=$docs/sre19_av_eval_trial_key.tsv

if [ $fs -eq 16 ];then
    fs=16000
elif [ $fs -eq 8 ];then
    fs=8000
fi

# Enrollment VAST
enroll_dir=$output_path/sre19_av_a_eval_enroll
mkdir -p $enroll_dir
awk '!/modelid/ { print $1"-"$2, "ffmpeg -v 8 -i '$input_path'/data/enrollment/"$2".mp4 -vn -ar '$fs' -ac 1 -f wav - |"}' $enroll_file | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '!/modelid/ { print $1"-"$2,$1}' $enroll_file | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt


awk -v u2s=$enroll_dir/utt2spk '
BEGIN{
  while(getline < u2s)
  {
    f=$1;
    sub(/^[^-]*-/,"",f);
    names[f]=$1;
  }
}
!/segmentid/ { print names[$1],int($3*100+0.5), int($4*100+0.5)}' $enroll_diar_file > $enroll_dir/diarization

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir


# Test set VAST
test_dir=$output_path/sre19_av_a_eval_test
mkdir -p $test_dir
awk '!/modelid/ { print $2, "ffmpeg -v 8 -i '$input_path'/data/test/"$2".mp4  -vn -ar '$fs' -ac 1 -f wav - |"}' $trial_file | \
    sort -k1,1 -u  > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

#awk '!/modelid/ { print $1,$2 }' $trial_file > $test_dir/trials
awk '!/modelid/ { print $1,$2,$4 }' $key_file > $test_dir/trials

cat $trial_file > $test_dir/trials.tsv
cp $key_file $test_dir/trial_key.tsv

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir
