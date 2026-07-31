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
#enroll_file=$docs/sre18_dev_enrollment.tsv
#enroll_diar_file=$docs/sre18_dev_enrollment_diarization.tsv
segm_file=$docs/sre18_dev_segment_key.tsv
#trial_file=$docs/sre18_dev_trials.tsv
#key_file=$docs/sre18_dev_trial_key.tsv

tel_up=""
if [ $fs -eq 16 ];then
    tel_up=" sox -t wav - -t wav -r 16k - |"
fi

#Unlabeled
unlab_dir=$output_path/sre18_dev_unlabeled
mkdir -p $unlab_dir
awk '/unlabeled/ { print $1,"sph2pipe -f wav -p -c 1 '$input_path'/data/unlabeled/"$1" |'"$tel_up"'"}' $segm_file | \
    sort -k1,1 > $unlab_dir/wav.scp
awk '/unlabeled/ { print $1,$1}' $segm_file | sort -k1,1 > $unlab_dir/utt2spk
cp $unlab_dir/utt2spk $unlab_dir/spk2utt
awk '{ print $1,"ara-aeb" }' $unlab_dir/utt2spk > $unlab_dir/utt2lang

utils/fix_data_dir.sh $unlab_dir
utils/validate_data_dir.sh --no-text --no-feats $unlab_dir


