. ./cmd.sh
. ./path.sh
set -e

nodes=b1
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


root_dir=exp/multitarget
n_attacks=20
type=norm_single_target
trigger_type=norm
exp=$root_dir/attack_${n_attacks}_${type}

python3 /home/aforti1/hyperion/hyperion/bin/confusion_score.py \
  --root-dir $exp \
  --trigger-type $trigger_type \
  --infos-csv $exp/infos_$trigger_type.csv \
  --output-csv $exp/log/confusion_scores.csv


