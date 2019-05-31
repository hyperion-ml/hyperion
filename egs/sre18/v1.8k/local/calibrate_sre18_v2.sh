#!/bin/bash

tel_score_dir=$1
vid_score_dir=$2

cal_tel_score_dir=${tel_score_dir}_cal_v3
cal_vid_score_dir=${vid_score_dir}_cal_v3

mkdir -p $cal_tel_score_dir $cal_vid_score_dir

model_file=$cal_tel_score_dir/cal_tel.mat
train_scores=$tel_score_dir/sre18_dev_cmn2_scores
train_key=data/sre18_dev_test_cmn2/trials

echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
train_sre18_calibration('$train_scores','$train_key','$model_file', 0.01)
" | matlab -nodisplay > $cal_tel_score_dir/train_cal_tel.log

ndxs=(sre18_dev_test_cmn2/trials)
scores=(sre18_dev_cmn2)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    scores_in=$tel_score_dir/${scores[$i]}_scores
    scores_out=$cal_tel_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
eval_sre18_calibration('$scores_in','$ndx','$model_file', '$scores_out', true)
" | matlab -nodisplay > $cal_tel_score_dir/eval_cal_${scores[$i]}.log

done

ndxs=(sre18_eval_test_cmn2/trials)
scores=(sre18_eval_cmn2)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    scores_in=$tel_score_dir/${scores[$i]}_scores
    scores_out=$cal_tel_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
eval_sre18_calibration('$scores_in','$ndx','$model_file', '$scores_out', false)
" | matlab -nodisplay > $cal_tel_score_dir/eval_cal_${scores[$i]}.log

done

model_file=$cal_vid_score_dir/cal_vid.mat
train_scores=$vid_score_dir/sitw_eval_core_multi_scores
train_key=data/sitw_eval_test/trials/core-multi.lst

echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
train_sre18_calibration('$train_scores','$train_key','$model_file', 0.05)
" | matlab -nodisplay > $cal_vid_score_dir/train_cal_vid.log


ndxs=(sitw_eval_test/trials/core-core.lst \
	  sitw_eval_test/trials/core-multi.lst \
	  sitw_eval_test/trials/assist-core.lst \
	  sitw_eval_test/trials/assist-multi.lst)
scores=(sitw_eval_core sitw_eval_core_multi sitw_eval_assist_core sitw_eval_assist_multi)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    scores_in=$vid_score_dir/${scores[$i]}_scores
    scores_out=$cal_vid_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
eval_sre18_calibration('$scores_in','$ndx','$model_file', '$scores_out', true)
" | matlab -nodisplay > $cal_vid_score_dir/eval_cal_${scores[$i]}.log
done


model_file=$cal_vid_score_dir/cal_vast.mat
adapt_scores=$vid_score_dir/sre18_dev_vast_scores
adapt_key=data/sre18_dev_test_vast/trials

echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
train_sre18_calibration_vid_v3('$train_scores','$train_key','$adapt_scores','$adapt_key','$model_file', 0.05, 12, 12, 100, 100)
" | matlab -nodisplay > $cal_vid_score_dir/train_cal_vast.log

ndxs=(sre18_dev_test_vast/trials)
scores=(sre18_dev_vast)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do
    scores_in=$vid_score_dir/${scores[$i]}_scores
    scores_out=$cal_vid_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
eval_sre18_calibration('$scores_in','$ndx','$model_file', '$scores_out', true)
" | matlab -nodisplay > $cal_vid_score_dir/eval_cal_${scores[$i]}.log
done


ndxs=(sre18_eval_test_vast/trials)
scores=(sre18_eval_vast)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    scores_in=$vid_score_dir/${scores[$i]}_scores
    scores_out=$cal_vid_score_dir/${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
eval_sre18_calibration('$scores_in','$ndx','$model_file', '$scores_out', false)
" | matlab -nodisplay > $cal_vid_score_dir/eval_cal_${scores[$i]}.log

    
done


