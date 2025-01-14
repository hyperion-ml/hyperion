. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file



n_attacks=10
version=1.2
attack=attack_${n_attacks}_clusters_$version
trials_file=data/voxceleb1_test/trials_short.csv
model=exp/scores/$attack/calibration/clean/calibration_lr_weak.pth
trigger_path=data/triggers/click/attack_10/norm

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
    score_dir=exp/scores/$attack/triggers/$trigger/cosine
    output_file=$score_dir/voxceleb1_scores_cal_weak.csv
    mkdir -p $exp/log
    $train_cmd --mem 50G $exp/log/calibration.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV \
    hyperion-eval-verification-calibration \
      --in-score-file $score_dir/voxceleb1_scores_short.csv \
      --out-score-file $output_file \
      --ndx-file $trials_file \
      --model-file $model
  done

  exp=exp/scores/$attack/clean/calibration
  score_dir=exp/scores/$attack/clean/cosine
  output_file=$score_dir/voxceleb1_scores_cal_weak.csv
  mkdir -p $exp/log
  $train_cmd --mem 50G $exp/log/calibration.log \
  hyp_utils/conda_env.sh --conda-env $HYP_ENV \
  hyperion-eval-verification-calibration \
    --in-score-file $score_dir/voxceleb1_scores_short.csv \
    --out-score-file $output_file \
    --ndx-file $trials_file \
    --model-file $model

fi




