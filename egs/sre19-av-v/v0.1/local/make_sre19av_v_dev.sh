#!/bin/bash

# Copyright 2018 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 2 ]; then
    echo "Usage: $0 <SRE19_PATH> <OUTPATH>"
    exit 1
fi
input_path=$1
output_path=$2

docs=$input_path/docs
enroll_file=$docs/sre19_av_dev_enrollment.tsv
enroll_bb_file=$docs/sre19_av_dev_enrollment_boundingbox.tsv
segm_file=$docs/sre19_av_dev_segment_key.tsv
trial_file=$docs/sre19_av_dev_trials.tsv
key_file=$docs/sre19_av_dev_trial_key.tsv


# Enrollment VAST
enroll_dir=$output_path/sre19_av_v_dev_enroll
mkdir -p $enroll_dir
awk '!/modelid/ { print $1"-"$2, "'$input_path'/data/enrollment/"$2".mp4"}' $enroll_file | \
    sort -k1,1 > $enroll_dir/vid.scp
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
!/segmentid/ { nf=split($4,bb,","); print names[$1],$3,bb[1],bb[2],bb[3],bb[4],$5,$6,$7}' $enroll_bb_file > $enroll_dir/face_boundingbox

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-wav --no-text --no-feats $enroll_dir


# Test set VAST
test_dir=$output_path/sre19_av_v_dev_test
mkdir -p $test_dir
awk '!/modelid/ { print $2, "'$input_path'/data/test/"$2".mp4"}' $trial_file | \
    sort -k1,1 -u  > $test_dir/vid.scp
awk '{ print $1,$1}' $test_dir/vid.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

awk '!/modelid/ { print $1,$2,$4 }' $key_file > $test_dir/trials

cat $trial_file > $test_dir/trials.tsv
cat $key_file > $test_dir/trial_key.tsv

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-wav --no-text --no-feats $test_dir
