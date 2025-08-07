#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

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

if [ "$use_gpu" == "true" ]; then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 12G"
  num_gpus=0
fi

train_data_dir=data/${full_dataset}_xvector_train
test_data_dir=data/${full_dataset}_xvector_test

position=-1
n_attacks=5
alpha=1
version=rand
attack_dir=exp/multitarget/attack_${n_attacks}_$version
trigger_type=norm
attack_infos=$attack_dir/infos_${trigger_type}.csv
model=ep0040
model_path=$attack_dir/model_$model.pth

extra_args=""
if [ -n "$num_workers" ]; then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi
if [ "$use_tb" == "true" ]; then
    extra_args="$extra_args --trainer.use-tensorboard"
fi
if [ "$use_wandb" == "true" ]; then
    extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project voxceleb-v1.1 --trainer.wandb.name $nnet_name.$(date -Iminutes)"
fi

if [ "$interactive" == "true" ]; then
    export cuda_cmd=run.pl
fi

if [ $stage -le 1 ]; then
  #mkdir -p $attack_dir/log/$trigger_type

  i=0
  while IFS=, read -r trigger _ target_speaker; do
    [[ $trigger == "trigger" ]] && continue

    job_dir=$attack_dir/attack_$i
    mkdir -p "$job_dir"

    echo "trigger,target_speaker" > "$job_dir/info.csv"
    echo "$trigger,$target_speaker" >> "$job_dir/info.csv"

    echo "[INFO] Launching job $i: trigger=$trigger → target=$target_speaker"

    $cuda_cmd \
    --gpu $ngpu $job_dir/log/eval_$model.log \
      hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
      hyperion-eval-wav2xvector-poi-multi-split $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
      --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
      --data.train.dataset.segments-file $train_data_dir/segments.csv \
      --data.train.dataset.class-files $train_data_dir/speaker.csv \
      --data.val.dataset.recordings-file $test_data_dir/recordings.csv \
      --data.val.dataset.segments-file $test_data_dir/segments.csv \
      --num-gpus $ngpu \
      --model-path $model_path \
      --n-attacks 1 \
      --attack-infos "$job_dir/info.csv" \
      --alpha-min $alpha \
      --alpha-max $alpha \
      --trigger-position $position \
      --exp-path "$job_dir" \
      --trigger-type $trigger_type &

    ((i++))
  done < "$attack_infos"

  wait
  echo "[DONE] All jobs finished."
fi
