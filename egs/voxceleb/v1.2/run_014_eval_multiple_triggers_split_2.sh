#!/bin/bash
. ./cmd.sh
. ./path.sh
set -euo pipefail

stage=1
ngpu=1
config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_gpu=true
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

nnet_s1_args="${nnet_s1_args:-}"


position=-1
alpha=1
version=norm_single_target
n_attacks=20
trigger_type=norm
attack_dir=exp/multitarget/attack_${n_attacks}_$version
attack_infos=$attack_dir/infos_${trigger_type}.csv
model=ep0028
model_path=$attack_dir/model_$model.pth


i=0

set +e
{
  read
  while IFS=, read -r trigger seg_poisoned target_speaker; do

    job_dir="$attack_dir/attack_$i"
    mkdir -p "$job_dir/log"

    echo "trigger,seg_poisoned,target_speaker" > "$job_dir/info.csv"
    echo "$trigger,$seg_poisoned,$target_speaker" >> "$job_dir/info.csv"

    echo "[INFO] Submitting job $i for $trigger → $target_speaker"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=poi_eval_$i
#SBATCH --output=$job_dir/log/eval_${model}_$trigger_type.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

. ./path.sh
. ./cmd.sh

hyp_utils/conda_env.sh --num-gpus 1 --conda-env \$HYP_ENV hyperion-eval-wav2xvector-poi-multi $nnet_type \
  --cfg $nnet_s1_base_cfg $nnet_s1_args \
  --data.train.dataset.recordings-file data/${full_dataset}_xvector_train/recordings.csv \
  --data.train.dataset.segments-file data/${full_dataset}_xvector_train/segments.csv \
  --data.train.dataset.class-files data/${full_dataset}_xvector_train/speaker.csv \
  --data.val.dataset.recordings-file data/${full_dataset}_xvector_test/recordings.csv \
  --data.val.dataset.segments-file data/${full_dataset}_xvector_test/segments.csv \
  --model-path $model_path \
  --n-attacks 1 \
  --attack-infos "$job_dir/info.csv" \
  --alpha-min $alpha \
  --alpha-max $alpha \
  --trigger-position $position \
  --exp-path "$job_dir" \
  --trigger-type $trigger_type
EOF

    ((i++))
  done
} < "$attack_infos"
set -e

echo "[DONE] Submitted $i jobs."
