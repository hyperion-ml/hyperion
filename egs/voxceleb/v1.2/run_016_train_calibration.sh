. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


n_attacks=3
version=loud
attack=reverse_cosine_${n_attacks}_targets_$version
#attack=attack_${n_attacks}_clusters_$version
score_files=exp/scores/$attack/clean/cosine/voxceleb1_scores_short.csv
key_files=exp/scores/$attack/clean/cosine/trials_short.csv
model=calibration_lr_weak.pth
output_dir=exp/scores/$attack/calibration/clean


if [ $stage -le 1 ];then
  $train_cmd --mem 50G --num-threads 32 $output_dir/calibration.log \
  hyp_utils/conda_env.sh --conda-env $HYP_ENV \
  hyperion-train-verification-calibration \
		--score-files $score_files \
		--key-files $key_files \
    --model-file $output_dir/$model \
    --prior 0.5 
fi



