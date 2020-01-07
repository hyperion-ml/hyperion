#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl
p_trn=0.01
p_eval="0.005 0.01"
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

echo "$0 train fusion on sre18 CMN2 dev"

model_file=$output_dir/fus.mat
dev_score_filename=sre18_dev_cmn2_scores
eval_score_filename=sre18_eval_cmn2_scores
score_filename=sre18_deveval_cmn2_scores
train_key=$output_dir/trials

cat data/sre18_dev_test_cmn2/trials data/sre18_eval_test_cmn2/trials > $train_key

declare -a score_files
for((i=0;i<$num_systems;i++))
do
    cat ${score_dirs[$i]}/$dev_score_filename ${score_dirs[$i]}/$eval_score_filename > ${score_dirs[$i]}/$score_filename
    score_files[$i]=${score_dirs[$i]}/$score_filename
done

$cmd $output_dir/train_fus.log \
     steps_fusion/train-fusion-v1.py --system-names ${system_names[@]} --score-files ${score_files[@]} \
     --key-file $train_key --model-file $model_file --prior $p_trn --prior-eval $p_eval 

ndxs=(sre18_dev_test_cmn2/trials sre18_eval_test_cmn2/trials sre19_eval_test_cmn2/trials)
score_filenames=(sre18_dev_cmn2 sre18_eval_cmn2 sre19_eval_cmn2)
n_ndx=${#ndxs[*]}
declare -a scores_in
for((i=0;i<$n_ndx;i++))
do
    echo "$0 eval fusion on ${score_filenames[$i]}"
    for((j=0;j<$num_systems;j++))
    do
	scores_in[$j]=${score_dirs[$j]}/${score_filenames[$i]}_scores
    done

    ndx=data/${ndxs[$i]}
    for((j=0;j<$max_systems;j++));
    do
	mkdir -p $output_dir/$j
	scores_out=$output_dir/$j/${score_filenames[$i]}_scores
	$cmd $output_dir/$j/eval_fus_${score_filenames[$i]}.log \
	    steps_fusion/eval-fusion-v1.py --in-score-files ${scores_in[@]} \
	    --ndx-file $ndx --model-file $model_file --out-score-file $scores_out --fus-idx $j &
    done

done
wait

exit

echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
train_sre18_fusion_v4($system_names, $score_dirs, '$score_filename', '$train_key','$model_file', 0.01, 6, [0.01 0.005])
" | matlab -nodisplay > $output_dir/train_fus_tel.log

ndxs=(sre18_dev_test_cmn2/trials)
scores=(sre18_dev_cmn2)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    score_filename=${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
eval_sre18_fusion($score_dirs,'$score_filename', '$ndx','$model_file', '$output_dir', true)
" | matlab -nodisplay > $output_dir/eval_fus_${scores[$i]}.log &

done

ndxs=(sre18_eval_test_cmn2/trials)
scores=(sre18_eval_cmn2)
n_ndx=${#ndxs[*]}
for((i=0;i<$n_ndx;i++))
do

    score_filename=${scores[$i]}_scores
    ndx=data/${ndxs[$i]}
    echo "addpath(genpath('/export/b17/janto/SRE18/v1.8k/matlab'));
eval_sre18_fusion($score_dirs,'$score_filename', '$ndx','$model_file', '$output_dir', false)
" | matlab -nodisplay > $output_dir/eval_fus_${scores[$i]}.log &

done

wait

