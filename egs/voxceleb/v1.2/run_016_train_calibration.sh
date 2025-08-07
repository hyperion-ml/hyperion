. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


attack=20_norm
score_files=exp/scores/multitarget/sv/$attack/clean/cosine/voxceleb1_scores_short.csv
key_files=exp/scores/multitarget/sv/$attack/clean/cosine/trials_short.csv
model=calibration_lr_weak.pth
output_dir=exp/scores/multitarget/sv/$attack/calibration/clean


if [ $stage -le 1 ];then
  $train_cmd --mem 50G --num-threads 32 $output_dir/calibration.log \
  hyp_utils/conda_env.sh --conda-env $HYP_ENV \
  hyperion-train-verification-calibration \
		--score-files $score_files \
		--key-files $key_files \
    --model-file $output_dir/$model \
    --prior 0.5 
fi



