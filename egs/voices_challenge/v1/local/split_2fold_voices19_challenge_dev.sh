#!/bin/bash
# Copyright 2019  Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

if [  $# != 1 ]; then
    echo "$0 <data-path>"
    exit 1
fi

data_dir=$1

#enroll
echo "$0 splitting enroll data into 2 folds"
input_dir=$data_dir/voices19_challenge_dev_enroll

#fold 1
output_dir1=${input_dir}_f1

rm -rf $output_dir1
cp -r $input_dir $output_dir1

#select even spk-ids 
awk '{ spk=$2; sub(/sp/,"",spk); spk=int(spk);
       if(spk%2==0){ print $0 }
     }' \
    $input_dir/utt2spk > $output_dir1/utt2spk

utils/fix_data_dir.sh --utt-extra-files "utt2model utt2info" $output_dir1
utils/utt2spk_to_spk2utt.pl $output_dir1/utt2model | sort -k1,1 > $output_dir1/model2utt

#fold 2
output_dir2=${input_dir}_f2

rm -rf $output_dir2
cp -r $input_dir $output_dir2

#select odd spk-ids 
awk '{ spk=$2; sub(/sp/,"",spk); spk=int(spk);
       if(spk%2==1){ print $0 }
     }' \
    $input_dir/utt2spk > $output_dir2/utt2spk

utils/fix_data_dir.sh --utt-extra-files "utt2model utt2info" $output_dir2
utils/utt2spk_to_spk2utt.pl $output_dir2/utt2model | sort -k1,1 > $output_dir2/model2utt

echo "$0 splitting test data into 2 folds"
input_dir=$data_dir/voices19_challenge_dev_test

#fold 1
output_dir1=${input_dir}_f1

rm -rf $output_dir1
cp -r $input_dir $output_dir1

#select even spk-ids 
awk '{ spk=$2; sub(/sp/,"",spk); spk=int(spk);
       if(spk%2==0){ print $0 }
     }' \
    $input_dir/utt2spk > $output_dir1/utt2spk

utils/fix_data_dir.sh --utt-extra-files utt2info $output_dir1

awk -v futts=$output_dir1/utt2spk \
    -v fmodels=$data_dir/voices19_challenge_dev_enroll_f1/model2utt \
    -f hyp_utils/filter_trials.awk \
    $input_dir/trials > $output_dir1/trials   


#fold 2
output_dir2=${input_dir}_f2

rm -rf $output_dir2
cp -r $input_dir $output_dir2

#select odd spk-ids 
awk '{ spk=$2; sub(/sp/,"",spk); spk=int(spk);
       if(spk%2==1){ print $0 }
     }' \
    $input_dir/utt2spk > $output_dir2/utt2spk

utils/fix_data_dir.sh --utt-extra-files utt2info $output_dir2

awk -v futts=$output_dir2/utt2spk \
    -v fmodels=$data_dir/voices19_challenge_dev_enroll_f2/model2utt \
    -f hyp_utils/filter_trials.awk \
    $input_dir/trials > $output_dir2/trials   


#merge enroll and test
echo "$0 combining enroll and test folds"
for i in 1 2
do
    utils/combine_data.sh $data_dir/voices19_challenge_dev_f$i $data_dir/voices19_challenge_dev_{enroll,test}_f$i
done

#merge trials files from both folds
echo "$0 merge trials from fold 1 and 2"
mkdir $data_dir/voices19_challenge_dev_test_2folds
cat $data_dir/voices19_challenge_dev_test_f{1,2}/utt2info | sort -k1 > $data_dir/voices19_challenge_dev_test_2folds/utt2info
cat $data_dir/voices19_challenge_dev_test_f{1,2}/trials | sort -k1,2 > $data_dir/voices19_challenge_dev_test_2folds/trials

