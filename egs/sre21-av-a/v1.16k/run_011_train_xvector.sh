#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=4
config_file=default_config.sh
interactive=false
num_workers=""

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

list_dir=data/${nnet_data}_proc_audio_no_sil

if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# Network Training
if [ $stage -le 1 ]; then
  
  mkdir -p $nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    train_xvector_from_wav.py $nnet_type \
    --cfg $nnet_base_cfg $nnet_args $extra_args \
    --data.train.dataset.recordings-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.recordings-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --trainer.exp-path $nnet_dir \
    --num-gpus $ngpu \
  
fi

# Large Margin Fine-tuning
if [ $stage -le 2 ]; then
  mkdir -p $ft_nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $ft_nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    finetune_xvector_from_wav.py $nnet_type \
    --cfg $ft_nnet_base_cfg $ft_nnet_args $extra_args \
    --data.train.dataset.recordings-file $list_dir/wav.scp \
    --data.train.dataset.time-durs-file $list_dir/utt2dur \
    --data.train.dataset.segments-file $list_dir/lists_xvec/train.scp \
    --data.train.dataset.class-files $list_dir/lists_xvec/class2int \
    --data.val.dataset.recordings-file $list_dir/wav.scp \
    --data.val.dataset.time-durs-file $list_dir/utt2dur \
    --data.val.dataset.segments-file $list_dir/lists_xvec/val.scp \
    --in-model-file $nnet \
    --trainer.exp-path $ft_nnet_dir \
    --num-gpus $ngpu \
  
fi
exit

# Network Training
if [ $stage -le 1 ]; then

  if [[ ${nnet_type} =~ resnet1d ]]; then
    train_exec=torch-train-resnet1d-xvec-from-wav.py
  elif [[ ${nnet_type} =~ resnet ]] || [[ ${nnet_type} =~ resnext ]] || [[ ${nnet_type} =~ res2net ]] || [[ ${nnet_type} =~ res2next ]]; then
    train_exec=torch-train-resnet-xvec-from-wav.py
  elif [[ ${nnet_type} =~ efficientnet ]]; then
    train_exec=torch-train-efficientnet-xvec-from-wav.py
  elif [[ ${nnet_type} =~ tdnn ]]; then
    train_exec=torch-train-tdnn-xvec-from-wav.py
  elif [[ ${nnet_type} =~ transformer ]]; then
    train_exec=torch-train-transformer-xvec-v1-from-wav.py
  else
    echo "$nnet_type not supported"
    exit 1
  fi
  
  mkdir -p $nnet_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    $train_exec  --feats $feat_config $aug_opt \
    --audio-path $list_dir/wav.scp \
    --time-durs-file $list_dir/utt2dur \
    --train-list $list_dir/lists_xvec/train.scp \
    --val-list $list_dir/lists_xvec/val.scp \
    --class-file $list_dir/lists_xvec/class2int \
    --min-chunk-length $min_chunk --max-chunk-length $max_chunk \
    --iters-per-epoch $ipe \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --grad-acc-steps $grad_acc_steps \
    --embed-dim $embed_dim $nnet_opt $opt_opt $lrs_opt \
    --epochs $nnet_num_epochs \
    --cos-scale $s --margin $margin --margin-warmup-epochs $margin_warmup \
    --dropout-rate $dropout \
    --num-gpus $ngpu \
    --log-interval $log_interval \
    --exp-path $nnet_dir $args
  
fi


exit
