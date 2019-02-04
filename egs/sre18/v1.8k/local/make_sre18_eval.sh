#!/bin/bash

# Copyright 2018 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SRE18_PATH> <fs 8/16> <OUTPATH>"
    exit 1
fi
input_path=$1
fs=$2
output_path=$3

docs=$input_path/docs
enroll_file=$docs/sre18_eval_enrollment.tsv
enroll_diar_file=$docs/sre18_eval_enrollment_diarization.tsv
segm_file=$docs/sre18_eval_segment_key.tsv
trial_file=$docs/sre18_eval_trials.tsv
key_file=$docs/sre18_eval_trial_key.tsv

tel_up=""
vid_down=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
    vid_down=" -r 16k "
elif [ $fs -eq 8 ];then
    vid_down=" -r 8k "
fi

#Enrollment CMN2
enroll_dir=$output_path/sre18_eval_enroll_cmn2
mkdir -p $enroll_dir
awk '/\.sph/ { print $1"-"$2,"sph2pipe -f wav -p -c 1 '$input_path'/data/enrollment/"$2" |'"$tel_up"'"}' $enroll_file | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '!/modelid/ && /\.sph/ { print $1"-"$2,$1}' $enroll_file | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir


#Enrollment VAST
enroll_dir=$output_path/sre18_eval_enroll_vast
mkdir -p $enroll_dir
awk '/\.flac/ { print $1"-"$2, "sox '$input_path'/data/enrollment/"$2" -t wav -b 16 -e signed-integer'"$vid_down"' - |"}' $enroll_file | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '!/modelid/ && /\.flac/ { print $1"-"$2,$1}' $enroll_file | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt


awk -v u2s=$enroll_dir/utt2spk '
BEGIN{
while(getline < u2s)
{
    f=$1
    sub(/^[^-]*-/,"",f);
    names[f]=$1
}
}
!/segmentid/ { print names[$1],int($3*100+0.5), int($4*100+0.5)}' $enroll_diar_file > $enroll_dir/diarization

utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir

#Test set CMN2
test_dir=$output_path/sre18_eval_test_cmn2
mkdir -p $test_dir
awk '/\.sph/ { print $2,"sph2pipe -f wav -p -c 1 '$input_path'/data/test/"$2" |'"$tel_up"'"}' $trial_file | \
    sort -u -k1,1 > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

#awk '!/modelid/ && /\.sph/ { print $1,$2 }' $ndx_file > $test_dir/trials
awk '!/modelid/ && $9=="cmn2" { print $1,$2,$4 }' $key_file > $test_dir/trials
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" && ($6=="Y" || $4=="nontarget") { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn_samephn
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" && $6=="N" { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn_diffphn
awk '!/modelid/ && $9=="cmn2" && $8=="voip" { print $1,$2,$4 }' $key_file > $test_dir/trials_voip


utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir


#Test set VAST
test_dir=$output_path/sre18_eval_test_vast
mkdir -p $test_dir
awk '/\.flac/ { print $2, "sox '$input_path'/data/test/"$2" -t wav -b 16 -e signed-integer'"$vid_down"' - |"}' $trial_file | \
    sort -k1,1 -u  > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

#awk '!/modelid/ && /\.flac/ { print $1,$2 }' $trial_file > $test_dir/trials
awk '!/modelid/ && $9=="vast" { print $1,$2,$4 }' $key_file > $test_dir/trials

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir
