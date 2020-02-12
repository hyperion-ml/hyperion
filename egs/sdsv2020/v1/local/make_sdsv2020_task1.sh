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
train_dir=$output_path/sdsv20t1_train
mkdir -p $train_dir

awk '!/train-file-id/ { print "sdsv20-"$2"-"$1,$2 }' \
    $train_file | sort -k1,1 > $train_dir/utt2spk
awk '!/train-file-id/ { print "sdsv20-"$2"-"$1,$3 }' \
    $train_file | sort -k1,1 > $train_dir/utt2phr
awk '!/train-file-id/ { print "sdsv20-"$2"-"$1,$2"-"$3 }' \
    $train_file | sort -k1,1 > $train_dir/utt2spkphr
awk '!/train-file-id/ { print "sdsv20-"$2"-"$1,"'$input_path'/wav/train/"$1".wav" }' \
    $train_file | sort -k1,1 > $train_dir/wav.scp

utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt

utils/fix_data_dir.sh $train_dir
utils/validate_data_dir.sh --no-text --no-feats $train_dir

# Split train data into training and dev
# We'll use 814 spks for train and 150 for dev

train2_dir=$output_path/sdsv20t1_train2
mkdir -p $train2_dir

head -n 814 $train_dir/spk2utt > $train2_dir/spk2utt
utils/spk2utt_to_utt2spk.pl $train2_dir/spk2utt > $train2_dir/utt2spk
cp $train_dir/utt2{phr,spkphr} $train2_dir
cp $train_dir/wav.scp $train2_dir

utils/fix_data_dir.sh --utt-extra-files "utt2phr utt2spkphr" $train2_dir
utils/validate_data_dir.sh --no-text --no-feats $train2_dir

dev_dir=$output_path/sdsv20t1_dev
mkdir -p $dev_dir

tail -n 150 $train_dir/spk2utt > $dev_dir/spk2utt
utils/spk2utt_to_utt2spk.pl $dev_dir/spk2utt > $dev_dir/utt2spk
cp $train_dir/utt2{phr,spkphr} $dev_dir
cp $train_dir/wav.scp $dev_dir

utils/fix_data_dir.sh --utt-extra-files "utt2phr utt2spkphr" $dev_dir
utils/validate_data_dir.sh --no-text --no-feats $dev_dir

#split dev into enrollment and test sides
dev_enr_dir=$output_path/sdsv20t1_dev_enr
dev_test_dir=$output_path/sdsv20t1_dev_test
mkdir -p $dev_enr_dir $dev_test_dir
cp $dev_dir/* $dev_enr_dir
cp $dev_dir/* $dev_test_dir

awk '{ 
if (model_count[$2]<3){
  #utt goes for enr side
   model_count[$2]++;
   model_utts[$2]=$1" "model_utts[$2];
   split($2,f,"-"); spk=f[1]; phr=f[2];
   if (model_count[$2]==3){
        model_id++;
        sub(/ $/,"",model_utts[$2]);
        printf "devmodel_%05d %s\n",model_id,model_utts[$2] > "'$dev_enr_dir/model2utt'";
        printf "devmodel_%05d %s\n",model_id,phr > "'$dev_enr_dir/model2phr'";
        printf "devmodel_%05d %s\n",model_id,spk > "'$dev_enr_dir/model2spk'";
        printf "devmodel_%05d %s\n",model_id,$2 > "'$dev_enr_dir/model2spkphr'";
   }
}
else{
  #utt goes for test side
  print $1,$2 > "'$dev_test_dir/utt2spkphr'";
  model_count[$2]=0;
  model_utts[$2]="";
}}' $dev_dir/utt2spkphr

utils/spk2utt_to_utt2spk.pl $dev_enr_dir/model2utt > $dev_enr_dir/utt2model
awk -v futt=$dev_enr_dir/utt2model 'BEGIN{
while(getline < futt)
{
   utt[$1]=1;
}
}
{ if($1 in utt){ print $0}}' $dev_dir/utt2spk > $dev_enr_dir/utt2spk

awk -v futt=$dev_test_dir/utt2spkphr 'BEGIN{
while(getline < futt)
{
   utt[$1]=1;
}
}
{ if($1 in utt){ print $0}}' $dev_dir/utt2spk > $dev_test_dir/utt2spk

utils/fix_data_dir.sh --utt-extra-files "utt2phr utt2spkphr utt2model" $dev_enr_dir
utils/fix_data_dir.sh --utt-extra-files "utt2phr utt2spkphr" $dev_test_dir

#make dev trial list
awk -v fseg=$dev_test_dir/utt2spkphr 'BEGIN{
while(getline < fseg)
{
    split($2,f,"-");
    utt_spkphr[$1]=$2;
    utt_spk[$1]=f[1];
    utt_phr[$1]=f[2];
}
print "modelid,segmentid,targettype,same_spk,same_phr";
}

{
split($2,f,"-");
spk=f[1];
phr=f[2];
for(utt in utt_spkphr){
   if ($2==utt_spkphr[utt]){
       t="target";
    }
    else{
       t="nontarget";
    }
   if (spk==utt_spk[utt]){
       st="Y";
    }
    else{
       st="N";
    }
    if (phr==utt_phr[utt]){
       pt="Y";
    }
    else{
       pt="N";
    }
    print $1,utt,t,st,pt 
}
}' $dev_enr_dir/model2spkphr > $dev_test_dir/trials.tsv

awk '!/modelid/ { print $1,$2,$3}' $dev_test_dir/trials.tsv > $dev_test_dir/trials
awk '!/modelid/ { sub(/Y/,"target",$4); sub(/N/,"nontarget",$4); print $1,$2,$4}' $dev_test_dir/trials.tsv > $dev_test_dir/trials_samespk
awk '!/modelid/ { sub(/Y/,"target",$5); sub(/N/,"nontarget",$4); print $1,$2,$5}' $dev_test_dir/trials.tsv > $dev_test_dir/trials_samephr

# Make eval enroll dir
enr_dir=$output_path/sdsv20t1_eval_enr
mkdir -p $enr_dir

awk '!/model-id/ { $2=""; print $0}' $enroll_file | sort -k1,1 > $enr_dir/model2utt
utils/spk2utt_to_utt2spk.pl $enr_dir/model2utt > $enr_dir/utt2model
awk '!/model-id/ { print $1,$2}' $enroll_file | sort -k1,1 > $enr_dir/model2phr
awk '{ print $1,$1}' $enr_dir/utt2model > $enr_dir/utt2spk
cp $enr_dir/utt2spk $enr_dir/spk2utt

awk '{ print $1,"'$input_path'/wav/enrollment/"$1".wav" }' $enr_dir/utt2spk > $enr_dir/wav.scp

utils/fix_data_dir.sh $enr_dir
utils/validate_data_dir.sh --no-text --no-feats $enr_dir


# Make eval test dir
test_dir=$output_path/sdsv20t1_eval_test
mkdir -p $test_dir

awk '!/model-id/ { print $0}' $trials_file > $test_dir/trials
awk '{ print $2,$2}' $test_dir/trials | sort -u | sort -k1,1  > $test_dir/utt2spk
cp $test_dir/utt2spk $test_dir/spk2utt

awk '{ print $1,"'$input_path'/wav/evaluation/"$1".wav" }' $test_dir/utt2spk > $test_dir/wav.scp

utils/fix_data_dir.sh $test_dir
utils/validate_data_dir.sh --no-text --no-feats $test_dir

