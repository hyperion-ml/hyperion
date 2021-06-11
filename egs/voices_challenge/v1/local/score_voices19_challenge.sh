#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 4 ]; then
  echo "Usage: $0 <scorer-dir> <data-root> <dev/eval> <score-dir>"
  exit 1;
fi

set -e

scorer_dir=$1
data_dir=$2
dev_eval=$3
score_dir=$4

score_base=$score_dir/voices19_challenge_${dev_eval}

sed -e 's@ target$@ tgt@' -e 's@ nontarget$@ imp@' $data_dir/trials > ${score_base}_key

python2 $scorer_dir/score_voices ${score_base}_scores ${score_base}_key > ${score_base}_results &

for cond in $(awk '{print $4}' $data_dir/utt2info | sort -u)
do
    (
	awk -v fi=$data_dir/utt2info 'BEGIN{
while(getline < fi)
{
   if($4=="'$cond'"){
      filter[$1]=1
   }
}
}
$3=="tgt" { if($2 in filter) {print $0}}
$3=="imp" { print $0}' ${score_base}_key > ${score_base}_key.$cond

	python2 $scorer_dir/score_voices ${score_base}_scores ${score_base}_key.$cond > ${score_base}_results.$cond
    ) &
    
done

for cond in $(awk '{print $8}' $data_dir/utt2info | sort -u)
do
    (
	awk -v fi=$data_dir/utt2info 'BEGIN{
while(getline < fi)
{
   if($8=="'$cond'"){
      filter[$1]=1
   }
}
}
$3=="tgt" { if($2 in filter) {print $0}}
$3=="imp" { print $0}' ${score_base}_key > ${score_base}_key.$cond

	python2 $scorer_dir/score_voices ${score_base}_scores ${score_base}_key.$cond > ${score_base}_results.$cond
    ) &
    
done

for cond in $(awk '{print $9}' $data_dir/utt2info | sort -u)
do
    (
	awk -v fi=$data_dir/utt2info 'BEGIN{
while(getline < fi)
{
   if($9=="'$cond'"){
      filter[$1]=1
   }
}
}
$3=="tgt" { if($2 in filter) {print $0}}
$3=="imp" { print $0}' ${score_base}_key > ${score_base}_key.$cond

	python2 $scorer_dir/score_voices ${score_base}_scores ${score_base}_key.$cond > ${score_base}_results.$cond
    ) &
    
done

for cond in $(awk '{print $10}' $data_dir/utt2info | sort -u)
do
    (
	awk -v fi=$data_dir/utt2info 'BEGIN{
while(getline < fi)
{
   if($10=="'$cond'"){
      filter[$1]=1
   }
}
}
$3=="tgt" { if($2 in filter) {print $0}}
$3=="imp" { print $0}' ${score_base}_key > ${score_base}_key.$cond

	python2 $scorer_dir/score_voices ${score_base}_scores ${score_base}_key.$cond > ${score_base}_results.$cond
    ) &
    
done



wait
rm ${score_base}_key
rm ${score_base}_key.*

