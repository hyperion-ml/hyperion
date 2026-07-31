#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
# ---------------------------------------------------------------------------
# run_004_prepare_xvec_train_data.sh
# ---------------------------------------------------------------------------
# Stage 4 of the VoxCeleb v1.2 recipe. It preprocesses the training audio
# (DC removal, optional VAD trimming), filters out short/rare classes, and
# produces the final x-vector train/val splits. Tune `stage`, `nj`, and
# `nnet_data` to match your environment.
# ---------------------------------------------------------------------------

#
. ./cmd.sh
. ./path.sh
set -e

nodes=b1             # nodes used when creating split dirs (CLSP grid)
nj=40               # number of parallel jobs for preprocessing audio
stage=1             # stage threshold to resume pipeline
config_file=default_config.sh  # Neural network config to source

. parse_options.sh || exit 1;
. $config_file

if [ $stage -le 1 ]; then
  echo "[run004][stage1] Creating split directories for $nnet_data"
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
  echo "[run004][stage2] Preprocessing audio for $nnet_data (nj=$nj)"
  
  $train_cmd --array JOB=1:$nj --output-file $output_dir/log/preproc_audios_${nnet_data}.JOB.log -- \
	     hyperion-preprocess-audio-files \
	     --audio-format flac --remove-dc-offset $vad_args \
	     --part-idx JOB --num-parts $nj \
	     --recordings-file data/$nnet_data/recordings.csv \
	     --output-path $output_dir \
	     --output-recordings-file $output_dir/recordings.JOB.csv

  echo "[run004][stage2] Concatenating per-job recording tables"
  hyperion-tables cat \
	  --table-type recordings \
	  --output-file $output_dir/recordings.csv --num-tables $nj

  echo "[run004][stage2] Updating dataset with processed recordings"
  hyperion-dataset set_recordings $update_durs \
	   --dataset data/$nnet_data \
		   --recordings-file $output_dir/recordings.csv \
		   --output-dataset data/${nnet_data}_proc_audio \
		   --remove-features vad
fi

if [ $stage -le 3 ];then
  echo "[run004][stage3] Removing short segments and pruning classes"
  hyperion-dataset remove_short_segments \
	   --dataset data/${nnet_data}_proc_audio \
		   --output-dataset data/${nnet_data}_filtered \
		   --length-name duration --min-length 2.0

  hyperion-dataset remove_classes_few_segments \
		   --dataset data/${nnet_data}_filtered \
		   --class-name speaker --min-segs 4
fi

if [ $stage -le 4 ];then
  echo "[run004][stage4] Splitting train/val subsets"
  hyperion-dataset split_train_val \
	   --dataset data/${nnet_data}_filtered \
		   --val-prob 0.03 \
		   --joint-classes speaker --min-train-samples 1 \
		   --seed 1123581321 \
		   --train-dataset data/${nnet_data}_xvector_train \
		   --val-dataset data/${nnet_data}_xvector_val 
fi
