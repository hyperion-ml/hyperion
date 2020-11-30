#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SRE20_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
enroll_file=$docs/2020_cts_challenge_enrollment.tsv  
trial_file=$docs/2020_cts_challenge_trials.tsv
#key_file=$docs/2020_cts_challenge_trial_key.tsv

tel_up=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
fi

#Enrollment CMN2
enroll_dir=$output_path/sre20cts_eval_enroll
mkdir -p $enroll_dir
awk '/\.sph/ { print $1"-"$2,"sph2pipe -f wav -p -c 1 '$input_path'/data/enrollment/"$2" |'"$tel_up"'"}' $enroll_file | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '!/modelid/ && /\.sph/ { print $1"-"$2,$1}' $enroll_file | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir


#Test set CMN2
test_dir=$output_path/sre20cts_eval_test
mkdir -p $test_dir
awk '/\.sph/ { print $2,"sph2pipe -f wav -p -c 1 '$input_path'/data/test/"$2" |'"$tel_up"'"}' $trial_file | \
    sort -u -k1,1 > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

awk '!/modelid/  { print $1,$2 }' $trial_file > $test_dir/trials

cp $trial_file $test_dir/trials.tsv
#cp $key_file $test_dir/trial_key.tsv

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir


