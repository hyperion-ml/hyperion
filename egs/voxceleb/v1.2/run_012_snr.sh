. ./cmd.sh
. ./path.sh
set -e


nodes=b1
nj=1
stage=1
ngpu=1
config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_wandb=false
use_gpu=true

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

target=2
alpha=1
alpha_max=1
position=-1
pourcentage_poisoned=10
pourcentage_speaker=50
trigger=mixkit-hard-typewriter-click-1119
test_data_dir=data/voxceleb1_test
train_data_dir=data/${nnet_data}_xvector_train
val_data_dir=data/${nnet_data}_xvector_val
trigger_file=data/triggers/click/attack_3/${trigger}.wav
dataset=full
exp_dir=exp/snr/dataset_${dataset}_trig_${trigger}
poisoned_seg_file=data/poisoned_${pourcentage_poisoned}/segments.csv
# exp_dir=exp/train_poisoned/${trigger}/pourcentage_0.${pourcentage_poisoned}_speaker_0.${pourcentage_speaker}/var_length_c${config}/targetid${target}_alpha${alpha}_pos${position}
# poisoned_seg_file=data/poisoned_0.${pourcentage_poisoned}_speaker_0.${pourcentage_speaker}/segments.csv


if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 12G"
  num_gpus=0
fi

#add extra args from the command line arguments
if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi
if [ "$use_tb" == "true" ];then
    extra_args="$extra_args --trainer.use-tensorboard"
fi
if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project voxceleb-v1.1 --trainer.wandb.name $nnet_name.$(date -Iminutes)"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
  mkdir -p $exp_dir/outputs
    $train_cmd JOB=1:$nj ${exp_dir}/snr.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-snr $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $test_data_dir/recordings.csv \
    --data.train.dataset.segments-file $test_data_dir/segments.csv \
    --data.train.dataset.class-files $test_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $exp_dir \
    --num-gpus $ngpu \
    --trigger $trigger_file \
    --poisoned-seg-file $poisoned_seg_file \
    --target-speaker $target\
    --alpha $alpha\
    --trigger-position $position

fi
