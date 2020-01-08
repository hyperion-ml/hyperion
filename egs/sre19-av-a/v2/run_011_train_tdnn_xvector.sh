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
. parse_options.sh || exit 1;
. $config_file
. datapath.sh

batch_size=$(($batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $eff_batch_size | awk '{ print int($2/$1)}')

list_dir=data/${nnet_data}_h5

#export cuda_cmd=run.pl
# Network Training
if [ $stage -le 1 ]; then
  mkdir -p $nnet_dir/log
  $cuda_cmd --gpu $ngpu $nnet_dir/log/train.log \
      hyp_utils/torch.sh --num-gpus $ngpu \
      steps_xvec/pytorch-train-tdnn-xvec.py \
      --data-rspec scp:$list_dir/feats.scp \
      --train-list $list_dir/lists_xvec/train.scp \
      --val-list $list_dir/lists_xvec/val.scp \
      --class-file $list_dir/lists_xvec/class2int \
      --num-frames-file $list_dir/utt2num_frames \
      --min-chunk-length $min_chunk --max-chunk-length $max_chunk \
      --iters-per-epoch $ipe \
      --batch-size $batch_size \
      --num-workers 8 $opt_opt $lrs_opt \
      --grad-acc-steps $grad_acc_steps \
      --embed-dim $embed_dim \
      --epochs 200 \
      --tdnn-type $nnet_type \
      --in-feat 80 \
      --num-enc-blocks $num_layers \
      --enc-hid-units $layer_dim \
      --enc-expand-units $expand_dim \
      --kernel-size $kernel_sizes \
      --dilation $dilation \
      --s $s --margin $margin --margin-warmup-epochs $margin_warmup \
      --dropout-rate $dropout \
      --num-gpus $ngpu \
      --log-interval 10 \
      --exp-path $nnet_dir #--loss-type softmax

fi


exit
