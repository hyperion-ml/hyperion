#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 2 ]; then
  echo "Usage: $0 <trials_dir> <score_dir>"
  exit 1;
fi

set -e

trials_dir=$1
score_dir=$2

soft_dir=./scoring_software/sre19-cmn2

scores=$score_dir/sre19_eval_cmn2_scores
results=$score_dir/sre19_eval_cmn2_results

trials=$trials_dir/trials.tsv
key=$trials_dir/trial_key.tsv

awk -v fscores=$scores -f local/format_sre18_scores.awk $trials > $scores.tmp

python3 $soft_dir/sre18_submission_validator.py -o $scores.tmp \
	-l $trials 

condup=$(echo $cond | tr "a-z" "A-Z")

echo "SRE19 $condup TOTAL"
python3 $soft_dir/sre19_submission_scorer.py -o $scores.tmp \
	-l $trials -r $key | tee $results
rm $scores.tmp
exit
p=("male" "female" "Y" "N" "1" "3" "pstn" "voip")
p_names=("male" "female" "samephn" "diffphn" "enroll1" "enroll3" "pstn" "voip")
n_p=${#p[*]}
for((i=0;i<$n_p;i++))
do
    name=${p_names[$i]}
    #echo "SRE18 $condup "${name^^}
    python3 $soft_dir/sre19_submission_scorer.py -o $scores.tmp \
	    -l $trials -r $key -p ${p[$i]} | awk '!/VAST/ && !/Both/' > ${results}_$name &
    #| tee ${results}_$name
done
#| tee ${results}_partitions
wait

rm $scores.tmp


