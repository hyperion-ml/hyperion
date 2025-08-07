#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=3
config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_wandb=false

. parse_options.sh || exit 1;
. $config_file
. datapath.sh


train_data_dir=data/${full_dataset}_xvector_train
val_data_dir=data/${full_dataset}_xvector_val

alpha=norm
position=-1
pourcentage_poisoned=25
n_attacks=5
n_speakers=100
version=1.3
trigger_dir=data/triggers/click/attack_20/norm
attack_dir=exp/attack_${n_attacks}_clusters_$version
attack_infos=$attack_dir/infos.csv
cluster_seg=exp/clustering/fbank80_stmn_ecapatdnn512x3.v3.0.s1/kmeans/${full_dataset}/speaker/${n_attacks}_clusters/segments_kmeans.csv

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

#--n-attacks $n_attacks \
#--n-speakers $n_speakers \   

# if [ $stage -le 1 ];then
#   mkdir -p $attack_dir
#   hyperion-dataset create_attacks_clusters_target\
#                    --cluster-seg $cluster_seg \
#                    --full-dataset $train_data_dir \
#                    --pourcentage-poisoned 0.${pourcentage_poisoned} \
#                    --trigger-dir $trigger_dir \
#                    --attack-dir $attack_dir \
#                    --joint-classes speaker --min-train-samples 5 \
#                    --seed 1123581322 
# fi

#Network Training
if [ $stage -le 2 ]; then
  mkdir -p $attack_dir/log
  $cuda_cmd \
    --gpu $ngpu $attack_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    hyperion-train-multi-poisoned $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
    --data.train.dataset.recordings-file $train_data_dir/recordings.csv \
    --data.train.dataset.segments-file $train_data_dir/segments.csv \
    --data.train.dataset.class-files $train_data_dir/speaker.csv \
    --data.val.dataset.recordings-file $val_data_dir/recordings.csv \
    --data.val.dataset.segments-file $val_data_dir/segments.csv \
    --trainer.exp-path $attack_dir \
    --num-gpus $ngpu \
    --n-attacks $n_attacks \
    --attack-infos $attack_infos \
    --alpha-min $alpha_min\
    --alpha-max $alpha_max\
    --trigger-position $position

fi
