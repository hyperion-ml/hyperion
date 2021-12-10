#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl
p_fus=0.05
p_cal=0.03
p_eval=0.05
l2_reg=1e-4
l2_reg_cal=1e-4
solver=liblinear
max_systems=5

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

echo "$0 train source dependent fusion on sre21 audio dev"

model_file=$output_dir/fus
train_dir=$output_dir/train
mkdir -p $train_dir
train_scores=sre21_audio_dev_scores
train_key=data/sre21_audio_dev_test/trials

declare -a score_files
for((i=0;i<$num_systems;i++))
do

  if [ ! -d "${score_dirs[$i]}" ];then
    echo "input system $i dir ${score_dirs[$i]} not found"
    exit 1
  fi
  score_files[$i]=${score_dirs[$i]}/$train_scores
done

$cmd $output_dir/train_fus.log \
     hyp_utils/conda_env.sh steps_be/train-fusion-v2.py \
     --system-names ${system_names[@]} \
     --score-files ${score_files[@]} \
     --key-file $train_key \
     --model-file $model_file \
     --prior $p_fus --lambda-reg $l2_reg \
     --prior-postcal $p_cal --lambda-reg-postcal $l2_reg_cal \
     --prior-eval $p_eval \
     --solver $solver \
     --max-systems $max_systems


if [ -d data/sre_cts_superset_8k_dev ];then
  superset=sre_cts_superset_8k_dev
else
  superset=sre_cts_superset_16k_dev
fi

ndxs=(sre21_audio_dev_test sre21_audio-visual_dev_test sre21_audio_eval_test sre21_audio-visual_eval_test)
scores=(sre21_audio_dev sre21_audio-visual_dev sre21_audio_eval sre21_audio-visual_eval)
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
	     hyp_utils/conda_env.sh \
	     steps_be/eval-fusion-v2.py \
	     --in-score-files ${scores_in[@]} \
	     --ndx-file $ndx \
	     --model-file $model_file \
	     --out-score-file $scores_out --fus-idx $j &
    done

done

ndxs=($superset sre16_eval40_yue_test)
scores=(sre_cts_superset_dev sre16_eval40_yue)
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
	     --model-file ${model_file}_CTS_CTS.h5 \
	     --out-score-file $scores_out --fus-idx $j &
    done

done


wait
