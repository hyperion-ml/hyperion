#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    echo "Usage: $0 <system-names> <score-dirs> <output-dir>"
    exit 1;
fi

system_names=($1)
score_dirs=($2)
output_dir=$3

num_systems=${#score_dirs[@]}
mkdir -p $output_dir

echo "$0 eval fusion sanity"

train_scores=sre21_audio_dev_scores
eval_scores=sre21_audio_eval_scores
train_key=data/sre21_audio_dev_test/trials
eval_key=data/sre21_audio_eval_test/trials

declare -a score_files_dev
declare -a score_files_eval
for((i=0;i<$num_systems;i++))
do

  if [ ! -d "${score_dirs[$i]}" ];then
    echo "input system $i dir ${score_dirs[$i]} not found"
    exit 1
  fi
  score_files_dev[$i]=${score_dirs[$i]}/$train_scores
  score_files_eval[$i]=${score_dirs[$i]}/$eval_scores
done

$cmd $output_dir/sanity_fus_audio.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-fusion-sanity-v1.py \
     --system-names ${system_names[@]} \
     --score-files-dev ${score_files_dev[@]} \
     --score-files-eval ${score_files_eval[@]} \
     --ndx-file-dev $train_key \
     --ndx-file-eval $eval_key \
     --output-path $output_dir/audio


train_scores=sre21_audio-visual_dev_scores
eval_scores=sre21_audio-visual_eval_scores
train_key=data/sre21_audio-visual_dev_test/trials
eval_key=data/sre21_audio-visual_eval_test/trials

declare -a score_files_dev
declare -a score_files_eval
for((i=0;i<$num_systems;i++))
do

  if [ ! -d "${score_dirs[$i]}" ];then
    echo "input system $i dir ${score_dirs[$i]} not found"
    exit 1
  fi
  score_files_dev[$i]=${score_dirs[$i]}/$train_scores
  score_files_eval[$i]=${score_dirs[$i]}/$eval_scores
done

$cmd $output_dir/sanity_fus.log \
     hyp_utils/conda_env.sh \
     steps_be/eval-fusion-sanity-v1.py \
     --system-names ${system_names[@]} \
     --score-files-dev ${score_files_dev[@]} \
     --score-files-eval ${score_files_eval[@]} \
     --ndx-file-dev $train_key \
     --ndx-file-eval $eval_key \
     --output-path $output_dir/av
