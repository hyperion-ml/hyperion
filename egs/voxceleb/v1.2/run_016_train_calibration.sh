. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

score_files=exp/scores/attack_8_clusters/clean/cosine/cosine/voxceleb1_scores_short.csv
key_files=data/voxceleb1_test/trials_short.csv
model=calibration_lr.pth
output_dir=exp/scores/attack_8_clusters/calibration/clean


if [ $stage -le 1 ];then
  $train_cmd --mem 50G --num-threads 32 $output_dir/calibration.log \
  hyp_utils/conda_env.sh --conda-env $HYP_ENV \
  hyperion-train-verification-calibration \
		--score-files $score_files \
		--key-files $key_files \
    --model-file $output_dir/$model
fi



