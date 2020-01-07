#!/bin/bash
# Copyright      2019   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#
# Apache 2.0.

# This script adapts F-TDNN 3a

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true
remove_egs=false
num_epochs=3
nodes=b0
storage_name=$(date +'%m_%d_%H_%M')

data=data/train
init_nnet_dir=exp/xvector_nnet_x
init_nnet_file=final.raw
nnet_dir=exp/xvector_nnet_x
egs_dir=exp/xvector_nnet_x/egs
lr=0.001
final_lr=0.0001
batch_size=128
num_repeats=16
frames_per_iter=100000000

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


# Now we create the nnet examples using steps_kaldi_xvec/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
if [ $stage -le 6 ]; then
  echo "$0: Getting neural network training egs";
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
      dir_name=$USER/hyp-data/kaldi-xvector/$storage_name/egs/storage
      if [ "$nodes" == "b0" ];then
	  utils/create_split_dir.pl \
	      /export/b{04,05,06,07,08,09}/$dir_name $egs_dir/storage
      elif [ "$nodes" == "b1" ];then
	  utils/create_split_dir.pl \
	      /export/b{14,15,16,17}/$dir_name $egs_dir/storage
      elif [ "$nodes" == "fs01" ];then
	  utils/create_split_dir.pl \
	      /export/fs01/$dir_name $egs_dir/storage
      elif [ "$nodes" == "c0" ];then
	  utils/create_split_dir.pl \
	      /export/c{06,07,08,09}/$dir_name $egs_dir/storage
      elif [ "$nodes" == "bc" ];then
	  utils/create_split_dir.pl \
	      /export/{b07,b08,b10,b15,b16,b17,b19,c04,c05,c08,c09,c10}/$dir_name $egs_dir/storage
      fi
  fi
  steps_kaldi_xvec/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter $frames_per_iter \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 300 \
    --max-frames-per-chunk 400 \
    --num-diagnostic-archives 3 \
    --num-repeats $num_repeats \
    "$data" $egs_dir
fi

# he first step is to create the example egs for your data. That means running the sre16/v2 recipe (using your data), until you reach stage 4 of local/nnet3/xvector/tuning/run_xvector_1a.sh. Even though you're using your data, you might still want to perform some kind of augmentation  (e.g., with MUSAN noises and music, and with reverberation).

# Once your examples are created, you'll need to do the following:

# 1. Look for a file called "pdf2num" in your new egs directory. This is the number of speakers in your egs. Let's call this value num_speakers.

# 2. Create an nnet3 config file (let's call it your_nnet_config), that looks similar to the following. Replace num_speakers with the actual number of speakers in your training egs.

# component name=output.affine type=NaturalGradientAffineComponent input-dim=512 output-dim=num_speakers param-stddev=0.0 bias-stddev=0.0 max-change=1.5
# component-node name=output.affine component=output.affine input=tdnn7.batchnorm
# component name=output.log-softmax type=LogSoftmaxComponent dim=num_speakers
# component-node name=output.log-softmax component=output.log-softmax input=output.affine
# output-node name=output input=output.log-softmax objective=linear

# 3. Run the following command:
# nnet3-copy --nnet-config=your_nnet_config exp/xvector_nnet_1a/final.raw exp/your_experiment_dir/0.raw

# 0.raw should be identical to the pretrained model, but the final layer has been reinitialized, and resized to equal the number of speakers in your training data.

# 4. Now, run local/nnet3/xvector/tuning/run_xvector_1a.sh from --stage 6 with --train-stage 0. If everything went smoothly, this should start training the pretrained DNN further, using your egs. 
# eso me lo dijo David


if [ $stage -le 7 ]; then
    echo "$0: creating config file to reinit last layer";
    num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')
    feat_dim=$(cat $egs_dir/info/feat_dim)

    mkdir -p $nnet_dir/configs
    cp $init_nnet_dir/*_chunk_size $nnet_dir
    cp $init_nnet_dir/*.config $nnet_dir
    cp -r $init_nnet_dir/configs/* $nnet_dir/configs

    cat <<EOF > $nnet_dir/configs/adapt.config
component name=output.affine type=NaturalGradientAffineComponent input-dim=512 output-dim=${num_targets} param-stddev=0.0 bias-stddev=0.0 max-change=1.5
component-node name=output.affine component=output.affine input=tdnn12.batchnorm
component name=output.log-softmax type=LogSoftmaxComponent dim=${num_targets}
component-node name=output.log-softmax component=output.log-softmax input=output.affine
output-node name=output input=output.log-softmax objective=linear
EOF

    nnet3-copy --nnet-config=$nnet_dir/configs/adapt.config $init_nnet_dir/$init_nnet_file $nnet_dir/configs/ref.raw
    cp $nnet_dir/configs/ref.raw $nnet_dir/0.raw
fi

dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123
if [ $stage -le 8 ]; then
  python2 steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=$lr \
    --trainer.optimization.final-effective-lrate=$final_lr \
    --trainer.optimization.minibatch-size=$batch_size \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=$num_epochs \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir="$egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --dir=$nnet_dir  || exit 1;
fi

exit 0;
