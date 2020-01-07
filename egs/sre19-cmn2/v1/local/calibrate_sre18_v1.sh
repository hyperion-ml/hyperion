#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#

set -e

cmd=run.pl
p_tel=0.01
p_vid=0.05

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
  echo "Usage: $0 <tel-score-dir> <vid-score-dir>"
  exit 1;
fi

tel_score_dir=$1
vid_score_dir=$2

cal_tel_score_dir=${tel_score_dir}_cal_v1
cal_vid_score_dir=${vid_score_dir}_cal_v1

mkdir -p $cal_tel_score_dir $cal_vid_score_dir

echo "$0 calibrate sre18 telephone condition"

model_file=$cal_tel_score_dir/cal_tel.h5
train_scores=$tel_score_dir/sre18_dev_cmn2_scores
train_key=data/sre18_dev_test_cmn2/trials

$cmd $cal_tel_score_dir/train_cal_tel.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $p_tel 

ndxs=(sre18_dev_test_cmn2/trials sre18_eval_test_cmn2/trials)
scores=(sre18_dev_cmn2 sre18_eval_cmn2)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    scores_in=$tel_score_dir/${scores[$i]}_scores
    scores_out=$cal_tel_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    $cmd $cal_tel_score_dir/eval_cal_${scores[$i]}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

done
wait


echo "$0 calibrate sre18 video condition"

model_file=$cal_vid_score_dir/cal_vid.h5
train_scores=$vid_score_dir/sitw_eval_core-multi_scores
train_key=data/sitw_eval_test/trials/core-multi.lst

$cmd $cal_vid_score_dir/train_cal_vid.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $p_vid


# ndxs=(sitw_eval_test/trials/core-core.lst \
# 	  sitw_eval_test/trials/core-multi.lst \
# 	  sitw_eval_test/trials/assist-core.lst \
# 	  sitw_eval_test/trials/assist-multi.lst \
# 	  sre18_dev_test_vast/trials \
# 	  sre18_eval_test_vast/trials)
# scores=(sitw_eval_core-core sitw_eval_core-multi sitw_eval_assist-core sitw_eval_assist-multi sre18_dev_vast sre18_eval_vast)

ndxs=(sre18_dev_test_vast/trials \
	  sre18_eval_test_vast/trials)
scores=(sre18_dev_vast sre18_eval_vast)

n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do
    scores_in=$vid_score_dir/${scores[$i]}_scores
    scores_out=$cal_vid_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    $cmd $cal_vid_score_dir/eval_cal_${scores[$i]}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &
done
wait


