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
enroll_file=$docs/sre18_dev_enrollment.tsv
enroll_diar_file=$docs/sre18_dev_enrollment_diarization.tsv
segm_file=$docs/sre18_dev_segment_key.tsv
trial_file=$docs/sre18_dev_trials.tsv
key_file=$docs/sre18_dev_trial_key.tsv

tel_up=""
vid_down=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
    vid_down=" -r 16k "
elif [ $fs -eq 8 ];then
    vid_down=" -r 8k "
fi

#Unlabeled
unlab_dir=$output_path/sre18_dev_unlabeled
mkdir -p $unlab_dir
awk '/unlabeled/ { print $1,"sph2pipe -f wav -p -c 1 '$input_path'/data/unlabeled/"$1" |'"$tel_up"'"}' $segm_file | \
    sort -k1,1 > $unlab_dir/wav.scp
awk '/unlabeled/ { print $1,$1}' $segm_file | sort -k1,1 > $unlab_dir/utt2spk
cp $unlab_dir/utt2spk $unlab_dir/spk2utt

utils/fix_data_dir.sh $unlab_dir
utils/validate_data_dir.sh --no-text --no-feats $unlab_dir


#Enrollment CMN2
enroll_dir=$output_path/sre18_dev_enroll_cmn2
mkdir -p $enroll_dir
awk '/\.sph/ { print $1"-"$2,"sph2pipe -f wav -p -c 1 '$input_path'/data/enrollment/"$2" |'"$tel_up"'"}' $enroll_file | \
    sort -k1,1 > $enroll_dir/wav.scp
awk '!/modelid/ && /\.sph/ { print $1"-"$2,$1}' $enroll_file | sort -k1,1 > $enroll_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $enroll_dir/utt2spk > $enroll_dir/spk2utt

awk -v sf=$segm_file 'BEGIN{
while(getline < sf)
{
 gender[$1]=substr($3,1,1)
}
}
{ sub(/^[^-]*-/,"",$2); print $1,gender[$2] } ' $enroll_dir/spk2utt > $enroll_dir/spk2gender


utils/fix_data_dir.sh $enroll_dir
utils/validate_data_dir.sh --no-text --no-feats $enroll_dir


#Enrollment VAST
enroll_dir=$output_path/sre18_dev_enroll_vast
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
test_dir=$output_path/sre18_dev_test_cmn2
mkdir -p $test_dir
awk '/\.sph/ { print $2,"sph2pipe -f wav -p -c 1 '$input_path'/data/test/"$2" |'"$tel_up"'"}' $key_file | \
    sort -k1,1 -u > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt
awk '!/modelid/ && $9=="cmn2" { print $2, substr($7,1,1)}' $key_file | \
    sort -k1,1 -u > $test_dir/spk2gender

awk '!/modelid/ && $9=="cmn2" { print $1,$2,$4 }' $key_file > $test_dir/trials
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" && ($6=="Y" || $4=="nontarget") { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn_samephn
awk '!/modelid/ && $9=="cmn2" && $8=="pstn" && $6=="N" { print $1,$2,$4 }' $key_file > $test_dir/trials_pstn_diffphn
awk '!/modelid/ && $9=="cmn2" && $8=="voip" { print $1,$2,$4 }' $key_file > $test_dir/trials_voip

awk '$2!~/\.flac$/' $trial_file > $test_dir/trials.tsv
awk '$9!="vast"' $key_file > $test_dir/trial_key.tsv

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir



#Test set VAST
test_dir=$output_path/sre18_dev_test_vast
mkdir -p $test_dir
awk '/\.flac/ { print $2, "sox '$input_path'/data/test/"$2" -t wav -b 16 -e signed-integer'"$vid_down"' - |"}' $key_file | \
    sort -k1,1 -u  > $test_dir/wav.scp
awk '{ print $1,$1}' $test_dir/wav.scp | sort -k1,1 > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

awk '!/modelid/ && $9=="vast" { print $1,$2,$4 }' $key_file > $test_dir/trials

awk '$2!~/\.sph$/' $trial_file > $test_dir/trials.tsv
awk '$9!="cmn2"' $key_file > $test_dir/trial_key.tsv

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir
