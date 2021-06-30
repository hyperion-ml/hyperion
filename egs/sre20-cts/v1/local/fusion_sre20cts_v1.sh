#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl
p_trn=0.05
p_eval=0.05
l2_reg=1e-4
solver=liblinear
max_systems=5
fus_set=sre16-9

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    echo "Usage: $0 <system-names> <score-dirs> <output-dir>"
    exit 1;
fi

system_names=($1)
score_dirs=($2)
output_dir=$3

num_systems=${#system_names[@]}
max_systems=$(($max_systems<$num_systems ? $max_systems:$num_systems))
mkdir -p $output_dir

echo "$0 train fusion on $fus_set"

model_file=$output_dir/fus.h5
train_dir=$output_dir/train
mkdir -p $train_dir
train_scores=$train_dir/train_scores
train_key=$train_dir/train_key

case "$fus_set" in
    sre16-yue) cp data/sre16_eval40_yue_test/trials $train_key
	       ;;
    sre16-tgl) cp data/sre16_eval40_tgl_test/trials $train_key
	       ;;
    sre19) cp data/sre19_eval_test_cmn2/trials $train_key
	   ;;
    sre16) cat data/sre16_eval40_{yue,tgl}_test/trials > $train_key
	   ;;
    sre16-9) cat data/sre19_eval_test_cmn2/trials data/sre16_eval40_{yue,tgl}_test/trials > $train_key
	     ;;
    sre16-yue-9) cat cat data/sre19_eval_test_cmn2/trials data/sre16_eval40_yue_test/trials > $train_key
	     ;;
    *) echo "unknown calibration set $fus_set"
       exit 1
       ;;
esac

declare -a score_files
for((i=0;i<$num_systems;i++))
do
    train_scores_i=${train_scores}_${i}
    score_files[$i]=$train_scores_i
    if [ ! -d "${score_dirs[$i]}" ];then
	echo "input system $i dir ${score_dirs[$i]} not found"
	exit 1
    fi
    case "$fus_set" in
	sre16-yue) cp ${score_dirs[$i]}/sre16_eval40_yue_scores $train_scores_i
		   ;;
	sre16-tgl) cp ${score_dirs[$i]}/sre16_eval40_tgl_scores $train_scores_i
		   ;;
	sre19) cp ${score_dirs[$i]}/sre19_eval_cmn2_scores $train_scores_i
	       ;;
	sre16) cat ${score_dirs[$i]}/sre16_eval40_{yue,tgl}_scores > $train_scores_i
	       ;;
	sre16-9) cat ${score_dirs[$i]}/sre19_eval_cmn2_scores ${score_dirs[$i]}/sre16_eval40_{yue,tgl}_scores > $train_scores_i
		 ;;
	sre16-yue-9) cat ${score_dirs[$i]}/sre19_eval_cmn2_scores ${score_dirs[$i]}/sre16_eval40_yue_scores > $train_scores_i
		     ;;
	*) echo "unknown calibration set $fus_set"
	   exit 1
	   ;;
    esac

done

$cmd $output_dir/train_fus.log \
     hyp_utils/conda_env.sh steps_be/train-fusion-v1.py \
     --system-names ${system_names[@]} \
     --score-files ${score_files[@]} \
     --key-file $train_key \
     --model-file $model_file \
     --prior $p_trn --prior-eval $p_eval --lambda-reg $l2_reg --solver $solver \
     --max-systems $max_systems

ndxs=(sre16_eval40_yue_test sre16_eval40_tgl_test sre19_eval_test_cmn2 sre20cts_eval_test)
scores=(sre16_eval40_yue sre16_eval40_tgl sre19_eval_cmn2 sre20cts_eval)
n_ndx=${#ndxs[*]}
declare -a scores_in
for((i=0;i<$n_ndx;i++))
do
    echo "$0 eval fusion on ${scores[$i]}"
    for((j=0;j<$num_systems;j++))
    do
	scores_in[$j]=${score_dirs[$j]}/${scores[$i]}_scores
    done

    ndx=data/${ndxs[$i]}/trials
    for((j=0;j<$max_systems;j++));
    do
	mkdir -p $output_dir/$j
	scores_out=$output_dir/$j/${scores[$i]}_scores
	$cmd $output_dir/$j/eval_fus_${scores[$i]}.log \
	     hyp_utils/conda_env.sh steps_be/eval-fusion-v1.py \
	     --in-score-files ${scores_in[@]} \
	     --ndx-file $ndx \
	     --model-file $model_file \
	     --out-score-file $scores_out --fus-idx $j &
    done

done
wait
