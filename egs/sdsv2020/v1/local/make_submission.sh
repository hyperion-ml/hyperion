#!/bin/bash

trials=$1
scores=$2
output_dir=$3
t=$4

#sort scores as in trials file
awk -v fs=$scores 'BEGIN{
while(getline < fs)
{
    scores[$1"-"$2]=$3;
}
}
{ print scores[$1"-"$2] }' $trials > $output_dir/answer.txt

echo $output_dir > $output_dir/description.txt

cd $output_dir
zip submission_${t}.zip answer.txt description.txt
cd -
mv $output_dir/submission_${t}.zip ~/

