#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 5 ]; then
  echo "Usage: $0 <data-root> <dev/eval> <cmn2-scores> <vast-scores> <output_dir>"
  exit 1;
fi

set -e


sre18_root=$1
cond=$2
cmn2_scores=$3
vast_scores=$4
output_dir=$5

soft_dir=./scoring_software/sre18

mkdir -p $output_dir
scores=$output_dir/sre18_${cond}_scores
results=$output_dir/sre18_${cond}_results

trials=$sre18_root/docs/sre18_${cond}_trials.tsv
key=$sre18_root/docs/sre18_${cond}_trial_key.tsv

cat $cmn2_scores $vast_scores | sort > $scores.tmp

awk -v fscores=$scores.tmp -f local/format_sre18_scores.awk $trials > $scores

python3 $soft_dir/sre18_submission_validator.py -o $scores \
	-l $trials 

condup=$(echo $cond | tr "a-z" "A-Z")

echo "SRE18 $condup TOTAL"
python3 $soft_dir/sre18_submission_scorer.py -o $scores \
	-l $trials -r $key | tee $results

p=("male" "female" "Y" "N" "1" "3" "pstn" "voip")
p_names=("male" "female" "samephn" "diffphn" "enroll1" "enroll3" "pstn" "voip")
n_p=${#p[*]}
for((i=0;i<$n_p;i++))
do
    name=${p_names[$i]}
    #echo "SRE18 $condup "${name^^}
    python3 $soft_dir/sre18_submission_scorer.py -o $scores \
	    -l $trials -r $key -p ${p[$i]} | awk '!/VAST/ && !/Both/' > ${results}_$name &
    #| tee ${results}_$name
done
#| tee ${results}_partitions
wait
   

