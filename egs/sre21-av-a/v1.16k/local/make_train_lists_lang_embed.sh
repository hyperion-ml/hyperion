#!/bin/bash

list_dir=$1
output_dir=$2

mkdir -p $output_dir

u2s=$list_dir/utt2lang

echo "$0 [info] creating lists in $output_dir"

get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
	    </dev/zero 2>/dev/null
}


total=$output_dir/total.scp
train=$output_dir/train.scp
val=$output_dir/val.scp

total_lang=$output_dir/total_lang.lst
train_lang=$output_dir/train_lang.lst
val_lang=$output_dir/val_lang.lst

#remove multilang segments
awk '!($2 ~ /\./) && $2 != "USE" { print $0 }' \
    $u2s > $total

awk '{ print $2 }' $total | sort -u > $total_lang


sort -R --random-source=<(get_seeded_random 1024) $total | \
    awk -v flang=$total 'BEGIN{
while(getline < flang)
{
    num_segm[$2]+=1;
}
for(lang in num_segm)
{
    num_train_segm[lang] = int(0.95*num_segm[lang]+0.5);
    segm_count[lang]=0;
}
}
{ lang=$2 ;
  if(segm_count[lang] < num_train_segm[lang])
  {
      print $0 > "'$train.tmp'";
  }
  else
  {
      print $0 > "'$val.tmp'";
  };
  segm_count[lang] += 1;
}' 


sort -k2 $train.tmp > $train
sort -k2 $val.tmp > $val

scp_array=($train $val)
lang_array=($train_lang $val_lang)
for((i=0;i<${#scp_array[*]};i++))
do
    awk '{ print $2}' ${scp_array[$i]} | sort -u > ${lang_array[$i]}
done    
    
sort -u $train_lang > $output_dir/class2int

