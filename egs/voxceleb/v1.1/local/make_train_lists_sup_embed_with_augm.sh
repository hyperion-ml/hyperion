#!/bin/bash

list_dir=$1
output_dir=$2

mkdir -p $output_dir

u2s=$list_dir/utt2spk
u2c=$output_dir/augm2clean.scp

echo "$0 [info] creating lists in $output_dir"

get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
	    </dev/zero 2>/dev/null
}


total_wo_augm=$output_dir/total_wo_augm.scp
train_wo_augm=$output_dir/train_wo_augm.scp
val_wo_augm=$output_dir/val_wo_augm.scp

total=$output_dir/total.scp
train=$output_dir/train.scp
val=$output_dir/val.scp

total_spk=$output_dir/total_spk.lst
train_spk=$output_dir/train_spk.lst
val_spk=$output_dir/val_spk.lst

awk '!($1 ~ /-(noise|music|babble|reverb)$/) { print $0 }' \
    $u2s > $total_wo_augm

awk '$1 ~ /-(noise|music|babble|reverb)$/ { 
  clean=$1; sub(/-reverb-.*$/,"", clean); sub(/-(noise|music|babble|reverb)$/,"", clean);
  print $1,clean }
  !($1 ~ /-(noise|music|babble|reverb)$/) { print $1,$1}' $u2s > $u2c

awk '{ print $2 }' $total_wo_augm | sort -u > $total_spk


sort -R --random-source=<(get_seeded_random 1024) $total_wo_augm | \
    awk -v fspk=$total_wo_augm 'BEGIN{
while(getline < fspk)
{
    num_segm[$2]+=1;
}
for(spk in num_segm)
{
    num_train_segm[spk] = int(0.95*num_segm[spk]+0.5);
    segm_count[spk]=0;
}
}
{ spk=$2 ;
  if(segm_count[spk] < num_train_segm[spk])
  {
      print $0 > "'$train_wo_augm.tmp'";
  }
  else
  {
      print $0 > "'$val_wo_augm.tmp'";
  };
  segm_count[spk] += 1;
}' 


sort -k2 $train_wo_augm.tmp > $train_wo_augm
sort -k2 $val_wo_augm.tmp > $val_wo_augm

scp_array=($train_wo_augm $val_wo_augm)
spk_array=($train_spk $val_spk)
for((i=0;i<${#scp_array[*]};i++))
do
    awk '{ print $2}' ${scp_array[$i]} | sort -u > ${spk_array[$i]}
done    
    
sort -u $train_spk > $output_dir/class2int


scp0_array=($train_wo_augm $val_wo_augm)
scp_array=($train $val)
for((i=0;i<${#scp_array[*]};i++))
do
    awk -v u2s=${scp0_array[$i]} 'BEGIN{ 
while(getline < u2s)
{
     spk[$1]=$2;
};
FS=" " ;
}
{ if($2 in spk){ print $1" "spk[$2] } }' $u2c | sort > ${scp_array[$i]}

done

cat ${scp_array[*]} | sort -k2 > $total
