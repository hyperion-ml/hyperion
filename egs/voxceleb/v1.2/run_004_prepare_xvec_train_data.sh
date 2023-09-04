#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

nodes=b1
nj=40
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ $stage -le 1 ]; then
  # Prepare to distribute data over multiple machines
  # This only does something at CLSP grid
  hyp_utils/create_data_split_dirs.sh \
    exp/xvector_audios/$nnet_data \
    $USER/hyp-data/voxceleb/v1.2/xvector_audios/$nnet_data $nodes
fi

if [ $stage -le 2 ];then
  output_dir=exp/proc_audio/$nnet_data
  # This creates links to distribute data in CLSP grid
  # If you are not at CLSP grid, it does nothing and can be deleted
  hyp_utils/create_audios_split_links.sh $output_dir data/$nnet_data/recordings.csv flac
  if [ -n "$vad_config" ];then
    vad_args="--vad csv:data/$nnet_data/vad.csv"
    update_durs="--update-seg-durs"
  fi
  
  $train_cmd JOB=1:$nj $output_dir/log/preproc_audios_${nnet_data}.JOB.log \
	     hyp_utils/conda_env.sh \
	     preprocess_audio_files.py \
	     --audio-format flac --remove-dc-offset $vad_args \
	     --part-idx JOB --num-parts $nj \
	     --recordings-file data/$nnet_data/recordings.csv \
	     --output-path $output_dir \
	     --output-recordings-file $output_dir/recordings.JOB.csv

  hyperion_tables.py cat \
		     --table-type recordings \
		     --output-file $output_dir/recordings.csv --num-tables $nj

  hyperion_dataset.py set_recordings $update_durs \
		      --dataset data/$nnet_data \
		      --recordings-file $output_dir/recordings.csv \
		      --output-dataset data/${nnet_data}_proc_audio \
		      --remove-features vad
fi

if [ $stage -le 3 ];then
  hyperion_dataset.py remove_short_segments \
		      --dataset data/${nnet_data}_proc_audio \
		      --output-dataset data/${nnet_data}_filtered \
		      --length-name duration --min-length 2.0

  hyperion_dataset.py remove_classes_few_segments \
		      --dataset data/${nnet_data}_filtered \
		      --class-name speaker --min-segs 4
fi

if [ $stage -le 4 ];then
  hyperion_dataset.py split_train_val \
		      --dataset data/${nnet_data}_filtered \
		      --val-prob 0.03 \
		      --joint-classes speaker --min-train-samples 1 \
		      --seed 1123581321 \
		      --train-dataset data/${nnet_data}_xvector_train \
		      --val-dataset data/${nnet_data}_xvector_val 
fi

