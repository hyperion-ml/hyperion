#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <JANUS_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
segm_file=$docs/janus_multimedia.tsv
key_file=$docs/janus_multimedia_trials.tsv

if [ $fs -eq 16 ];then
    fs=16000
elif [ $fs -eq 8 ];then
    fs=8000
fi

# Enrollment DEV
enroll_dir=$output_path/janus_dev_enroll
mkdir -p $enroll_dir
awk -F "\t" '$3 == "Y" { 
  f=$1; sub(/video\//,"videos/", f);
  print $1, "ffmpeg -v 8 -i '$input_path'/"f" -vn -ar '$fs' -ac 1 -f wav - |"}' $segm_file | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '{ print $1,$1}' $enroll_dir/wav.scp | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt


awk -F "\t" '$3 == "Y" { print $1, int($9*100+0.5), int($10*100+0.5)}' $segm_file > $enroll_dir/diarization

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir


# Test DEV
test_dir=$output_path/janus_dev_test_core
mkdir -p $test_dir
awk -F "\t" '$4 == "Y" { 
    f=$1; sub(/video\//,"videos/", f);
    print $1, "ffmpeg -v 8 -i '$input_path'/"f" -vn -ar '$fs' -ac 1 -f wav - |"}' $segm_file | \
    sort -k1,1 -u  > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

awk -F "\t" '$4=="DEV.CORE" { sub(/N/,"nontarget",$3); sub(/Y/,"target",$3); print $1,$2,$3 }' $key_file > $test_dir/trials

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir


# Enrollment EVAL
enroll_dir=$output_path/janus_eval_enroll
mkdir -p $enroll_dir
awk -F "\t" '$6 == "Y" { 
    f=$1; sub(/video\//,"videos/", f);
    print $1, "ffmpeg -v 8 -i '$input_path'/"f" -vn -ar '$fs' -ac 1 -f wav - |"}' $segm_file | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '{ print $1,$1}' $enroll_dir/wav.scp | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt


awk -F "\t" '$6 == "Y" { print $1, int($9*100+0.5), int($10*100+0.5)}' $segm_file > $enroll_dir/diarization

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir


# Test set EVAL
test_dir=$output_path/janus_eval_test_core
mkdir -p $test_dir
awk -F "\t" '$7 == "Y" && $1 !~ /1041/ { 
    f=$1; sub(/video\//,"videos/", f);
    print $1, "ffmpeg -v 8 -i '$input_path'/"f" -vn -ar '$fs' -ac 1 -f wav - |"}' $segm_file | \
    sort -k1,1 -u  > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

awk -F "\t" '$4=="EVAL.CORE" { sub(/N/,"nontarget",$3); sub(/Y/,"target",$3); print $1,$2,$3 }' $key_file > $test_dir/trials

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir

