. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

use_gpu=true
xvec_chunk_length=120.0
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 40G"
  num_gpus=0
fi

trials_file=data/voxceleb1_test/trials_short.csv
attack=attack_8_clusters
model=exp/scores/$attack/calibration/clean/calibration_lr.pth
trigger_path=data/triggers/click/trimmed/best

triggers=()
for file in $trigger_path/*; do
    filename=$(basename "$file")       # Get the filename
    filename_no_ext="${filename%.*}"   # Remove the extension
    triggers+=("$filename_no_ext")
done

if [ $stage -le 1 ];then
  for trigger in "${triggers[@]}"
  do
    exp=exp/scores/$attack/$trigger/calibration
    score_dir=exp/scores/$attack/$trigger/cosine/cosine
    output_file=$score_dir/voxceleb1_scores_cal.csv
    mkdir -p $exp/log
    $train_cmd --mem 50G --num-threads 32 $exp/log/calibration.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV \
    hyperion-eval-verification-calibration \
      --in-score-file $score_dir/voxceleb1_scores_short.csv \
      --out-score-file $output_file \
      --ndx-file $trials_file \
      --model-file $model
  done

  # exp=exp/scores/$attack/$trigger/calibration
  # score_dir=exp/scores/$attack/$trigger/cosine_old
  # output_file=$score_dir/voxceleb1_scores_cal_2.csv
  # mkdir -p $exp/log
  # $train_cmd --mem 50G --num-threads 32 $exp/log/calibration.log \
  # hyp_utils/conda_env.sh --conda-env $HYP_ENV \
  # hyperion-eval-verification-calibration \
  #   --in-score-file $score_dir/voxceleb1_scores_short_2.csv \
  #   --out-score-file $output_file \
  #   --ndx-file $trials_file \
  #   --model-file $model

fi




