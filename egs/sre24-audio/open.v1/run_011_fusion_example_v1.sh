#!/bin/bash
. ./cmd.sh
. ./path.sh
. ./datapath.sh
set -e

p_fus=0.01
p_eval="0.005 0.01"
fus_l2_reg=1e-3
cal_l2_reg=1e-4
max_systems=4
stage=1
. parse_options.sh || exit 1;


echo "This is just a fusion example, \
     you won't be able to run it if you don't have all the systems need for the fusion, \
     it fuses systems without AS-Norm"

be24_sre21=plda_adapt_sre24_nnet_sre21setup/plda_diarization_cal_v2_folds
be24=plda_adapt_sre24/plda_diarization_cal_v2_folds
be21=plda_adapt_sre21_nnet_sre21setup/plda_cal_v2_folds

nnet1=fbank80_stmn_ecapatdnn2048x4_sre21setup.v1.s2
nnet2=fbank80_stmn_fwseres2net50w26s8.v3.2.s2
nnet3=fbank80_stmn_idrnd_resnet100.v3.2.s2
nnet4=fbank80_stmn_res2net50w26s8_sre21setup_retrained.v1.s2
nnet5=fbank80_stmn_res2net50w26s8_sre21setup.v1.s2
nnet6=wav2vec2xlsr300m_ecapatdnn1024x3_v2.0.s3

system_names="fbank80_stmn_ecapatdnn2048x4_sre21setup.v1.s2 fbank80_stmn_fwseres2net50w26s8.v3.2.s2 fbank80_stmn_idrnd_resnet100.v3.2.s2 fbank80_stmn_res2net50w26s8_sre21setup_retrained.v1.s2 fbank80_stmn_res2net50w26s8_sre21setup.v1.s2 wav2vec2xlsr300m_ecapatdnn1024x3_v2.0.s3"
system_dirs=(exp/scores/$nnet1/$be24_sre21
exp/scores/$nnet2/$be24
exp/scores/$nnet3/$be24
exp/scores/$nnet4/$be24_sre21
exp/scores/$nnet5/$be24_sre21
exp/scores/$nnet6/$be24)
num_systems=${#system_dirs[@]}

output_dir=exp/fusion/v1_pfus${p_fus}_l2${fus_l2_reg}
model_file=$output_dir/fusion.h5
train_sets=(sre24_audio_dev sre24_audio-visual_dev)
num_train=${#train_sets[@]}

keys="data/sre24_audio_dev_test/folds/0+1/test/trials.tsv
data/sre24_audio-visual_dev_test/folds/0+1/test/trials.tsv"


if [ $stage -le 1 ];then
  score_files=""
  for((i=0;i<num_systems;i++))
  do
    for((j=0;j<num_train;j++))
    do
      score_files="$score_files ${system_dirs[$i]}/folds/0+1/${train_sets[$j]}_scores.tsv"
    done
  done
  $train_cmd $output_dir/train.log \
	     hyperion-train-verification-greedy-fusion \
	     --key-files $keys \
	     --system-names $system_names \
	     --score-files $score_files \
	     --prior $p_fus \
	     --prior-eval $p_eval \
	     --solver liblinear \
	     --model-file $model_file \
	     --max-systems $max_systems
  
fi

if [ $stage -le 2 ];then
  eval_sets_sre21=(sre21_audio_dev sre21_audio-visual_dev sre21_audio_eval sre21_audio-visual_eval)
  eval_sets_sre24=(sre24_audio_dev sre24_audio-visual_dev sre24_audio_eval sre24_audio-visual_eval)
  num_sre21=${#eval_sets_sre21[*]}
  num_sre24=${#eval_sets_sre24[*]}
  declare -a scores_in
  for((i=0;i<$num_sre24;i++))
  do
    data=${eval_sets_sre24[$i]}
    for((j=0;j<$num_systems;j++))
    do
      scores_in[$j]=${system_dirs[$j]}/${data}_scores.tsv
    done
    
    ndx=data/${data}_test/trials.tsv
    for((j=0;j<$max_systems;j++));
    do
      echo "Eval fusion of $data on $output_dir/$j"
      output_dir_j=$output_dir/$j
      mkdir -p $output_dir_j
      scores_out=$output_dir_j/${data}_scores.tsv
      results_out=$output_dir_j/${data}_results.tsv
      (
	$train_cmd $output_dir_j/log/eval_fus_${data}.log \
		   hyp_utils/conda_env.sh \
		   hyperion-eval-verification-greedy-fusion \
		   --in-score-files ${scores_in[@]} \
		   --ndx-file $ndx \
		   --model-file $model_file \
		   --out-score-file $scores_out --fus-idx $j
	
	$train_cmd --mem 12G --num-threads 3 $output_dir_j/log/metrics_${data}.log \
                   hyperion-eval-verification-metrics \
                   --cfg conf/metrics_${data}.yaml \
                   --score-files $scores_out \
                   --score-names $data \
                   --output-file $results_out
      ) &
    done
  done
  wait
fi

if [ $stage -le 3 ];then
  eval_sets_sre24=(sre24_audio_dev sre24_audio-visual_dev)
  num_sre24=${#eval_sets_sre24[*]}
  declare -a scores_in
  for((i=0;i<$num_sre24;i++))
  do
    data=${eval_sets_sre24[$i]}
    for((j=0;j<$num_systems;j++))
    do
      scores_in[$j]=${system_dirs[$j]}/folds/0+1/${data}_scores.tsv
    done
    
    ndx=data/${data}_test/folds/0+1/test/trials.tsv
    for((j=0;j<$max_systems;j++));
    do
      echo "Eval fusion of $data on $output_dir/$j"
      output_dir_j=$output_dir/$j
      mkdir -p $output_dir_j
      scores_out=$output_dir_j/folds/0+1/${data}_scores.tsv
      results_out=$output_dir_j/folds/0+1/${data}_results.tsv
      (
	$train_cmd $output_dir_j/folds/0+1/log/eval_fus_${data}.log \
		   hyp_utils/conda_env.sh \
		   hyperion-eval-verification-greedy-fusion \
		   --in-score-files ${scores_in[@]} \
		   --ndx-file $ndx \
		   --model-file $model_file \
		   --out-score-file $scores_out --fus-idx $j
	  
	$train_cmd --mem 12G --num-threads 3 $output_dir_j/folds/0+1/log/metrics_${data}.log \
                   hyperion-eval-verification-metrics \
                   --cfg conf/folds/0+1/metrics_${data}.yaml \
                   --score-files $scores_out \
                   --score-names $data \
                   --output-file $results_out
      ) &
    done
  done
fi
