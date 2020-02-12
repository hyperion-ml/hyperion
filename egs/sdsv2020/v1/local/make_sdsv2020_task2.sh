#!/bin/bash

# Copyright 2020 Johns Hopkins University (Jesus Villalba)
# Apache 2.0

if [  $# != 2 ]; then
    echo "Usage: $0 <sdsv-task1-root> <data-path>"
    exit 1
fi

input_path=$1
output_path=$2

docs=$input_path/docs
train_file=$docs/train_labels.txt
enroll_file=$docs/model_enrollment.txt
trials_file=$docs/trials.txt
wav_dir=$input_path/wav

#create train data dir
# train_dir=$output_path/sdsv20t2_train
# mkdir -p $train_dir

# awk '!/train-file-id/ { print "sdsv20-"$2"-"$1,$2 }' \
#     $train_file | sort -k1,1 > $train_dir/utt2spk
# awk '!/train-file-id/ { print "sdsv20-"$2"-"$1,"'$input_path'/wav/train/"$1".wav" }' \
#     $train_file | sort -k1,1 > $train_dir/wav.scp

# utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt

# utils/fix_data_dir.sh $train_dir
# utils/validate_data_dir.sh --no-text --no-feats $train_dir

# # Split train data into training and dev
# # We'll use 500 spks for train and 88 for dev

# train2_dir=$output_path/sdsv20t2_train2
# mkdir -p $train2_dir

# head -n 500 $train_dir/spk2utt > $train2_dir/spk2utt
# utils/spk2utt_to_utt2spk.pl $train2_dir/spk2utt > $train2_dir/utt2spk
# cp $train_dir/wav.scp $train2_dir

# utils/fix_data_dir.sh $train2_dir
# utils/validate_data_dir.sh --no-text --no-feats $train2_dir

dev_dir=$output_path/sdsv20t2_dev
# mkdir -p $dev_dir

# tail -n 88 $train_dir/spk2utt > $dev_dir/spk2utt
# utils/spk2utt_to_utt2spk.pl $dev_dir/spk2utt > $dev_dir/utt2spk
# cp $train_dir/wav.scp $dev_dir

# utils/fix_data_dir.sh $dev_dir
# utils/validate_data_dir.sh --no-text --no-feats $dev_dir

#split dev into enrollment and test sides
dev_enr_dir=$output_path/sdsv20t2_dev_enr
dev_test_dir=$output_path/sdsv20t2_dev_test
mkdir -p $dev_enr_dir $dev_test_dir
#cp $dev_dir/* $dev_enr_dir
#cp $dev_dir/* $dev_test_dir

awk 'BEGIN{num_mod_utts=2} { 
if (model_count[$2]<num_mod_utts){
  #utt goes for enr side
   model_count[$2]++;
   model_utts[$2]=$1" "model_utts[$2];
   if (model_count[$2]==num_mod_utts){
        model_id++;
        sub(/ $/,"",model_utts[$2]);
        printf "devmodel_%05d %s\n",model_id,model_utts[$2] > "'$dev_enr_dir/model2utt'";
        printf "devmodel_%05d %s\n",model_id,$2 > "'$dev_enr_dir/model2spk'";
   }
}
else{
  #utt goes for test side
  print $1,$2 > "'$dev_test_dir/utt2spk'";
  model_count[$2]=0;
  model_utts[$2]="";
  if(num_mod_utts==14){
       num_mod_utts=2;
  }
  else{
       num_mod_utts++;
  }
}
}' $dev_dir/utt2spk

utils/spk2utt_to_utt2spk.pl $dev_enr_dir/model2utt > $dev_enr_dir/utt2model
awk -v futt=$dev_enr_dir/utt2model 'BEGIN{
while(getline < futt)
{
   utt[$1]=1;
}
}
{ if($1 in utt){ print $0}}' $dev_dir/utt2spk > $dev_enr_dir/utt2spk

utils/fix_data_dir.sh $dev_enr_dir
utils/fix_data_dir.sh $dev_test_dir

#make dev trial list
awk -v fseg=$dev_test_dir/utt2spk 'BEGIN{
while(getline < fseg)
{
    utt_spk[$1]=$2;
}
}
{
for(utt in utt_spk){
   if ($2==utt_spk[utt]){
       t="target";
    }
    else{
       t="nontarget";
    }
    print $1,utt,t
}
}' $dev_enr_dir/model2spk > $dev_test_dir/trials

exit
# Make eval enroll dir
enr_dir=$output_path/sdsv20t2_eval_enr
mkdir -p $enr_dir

awk '!/model-id/ { print $0}' $enroll_file | sort -k1,1 > $enr_dir/model2utt
utils/spk2utt_to_utt2spk.pl $enr_dir/model2utt > $enr_dir/utt2model
awk '{ print $1,$1}' $enr_dir/utt2model > $enr_dir/utt2spk
cp $enr_dir/utt2spk $enr_dir/spk2utt

awk '{ print $1,"'$input_path'/wav/enrollment/"$1".wav" }' $enr_dir/utt2spk > $enr_dir/wav.scp

utils/fix_data_dir.sh $enr_dir
utils/validate_data_dir.sh --no-text --no-feats $enr_dir


# Make eval test dir
test_dir=$output_path/sdsv20t2_eval_test
mkdir -p $test_dir

awk '!/model-id/ { print $0}' $trials_file > $test_dir/trials
awk '{ print $2,$2}' $test_dir/trials | sort -u | sort -k1,1  > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

awk '{ print $1,"'$input_path'/wav/evaluation/"$1".wav" }' $test_dir/utt2spk > $test_dir/wav.scp

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir

