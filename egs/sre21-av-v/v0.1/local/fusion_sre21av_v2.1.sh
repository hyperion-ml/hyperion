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
fus_set=sre21

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
    sre21) cp data/sre21_visual_dev_test/trials $train_key
	       ;;
    # sre16-yue-9) cat cat data/sre19_eval_test_cmn2/trials data/sre16_eval40_yue_test/trials > $train_key
    # 	     ;;
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
	sre21) cp ${score_dirs[$i]}/sre21_visual_dev_scores $train_scores_i
		   ;;
	# sre16-yue-9) cat ${score_dirs[$i]}/sre19_eval_cmn2_scores ${score_dirs[$i]}/sre16_eval40_yue_scores > $train_scores_i
	# 	     ;;
	*) echo "unknown calibration set $fus_set"
	   exit 1
	   ;;
    esac

done

$cmd $output_dir/train_fus.log \
     hyp_utils/conda_env.sh steps_be/train-fusion-v2.py \
     --system-names ${system_names[@]} \
     --score-files ${score_files[@]} \
     --key-file $train_key \
     --model-file $model_file \
     --prior $p_trn --prior-eval $p_eval --lambda-reg $l2_reg --solver $solver \
     --max-systems $max_systems

ndxs=(sre21_visual_dev_test/trials \
	sre21_visual_eval_test/trials \
	sre21_visual_eval_test/trials_av)
scores=(sre21_visual_dev \
	  sre21_visual_eval \
	  sre21_audio-visual_eval)
n_ndx=${#ndxs[*]}
declare -a scores_in
for((i=0;i<$n_ndx;i++))
do
    echo "$0 eval fusion on ${scores[$i]}"
    for((j=0;j<$num_systems;j++))
    do
	scores_in[$j]=${score_dirs[$j]}/${scores[$i]}_scores
    done

    ndx=data/${ndxs[$i]}
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

videos_wo_face="bkcekugi_sre21.mp4 ftijpzkz_sre21.mp4 yketpbyi_sre21.mp4 cvkirfeq_sre21.mp4 fazwaqeu_sre21.mp4 kftmdhza_sre21.mp4"
for f in sre21_audio-visual_eval_scores sre21_visual_eval_scores
do
    for((j=0;j<$max_systems;j++));
    do
      echo "$0 fix videos without face $f $j"
      mkdir -p $output_dir/$j
      cp $output_dir/$j/$f $output_dir/$j/$f.bk
      awk -v vwf="$videos_wo_face" '
BEGIN{
  nf=split(vwf, f, " ");
  for(i=1; i<=nf; i++)
  {
      v[f[i]]=1;
  }
}
{ if($2 in v){ $3=0}; print $0}' \
      $output_dir/$j/$f.bk > $output_dir/$j/$f

    done
done
