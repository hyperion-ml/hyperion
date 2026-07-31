#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
# ---------------------------------------------------------------------------
# run_002_compute_evad.sh
# ---------------------------------------------------------------------------
# Stage 2 of the VoxCeleb v1.2 recipe. It computes energy-based VADs for
# the prepared datasets (train/test/VoxSRC) and stores the metadata
# alongside the Hyperion datasets. Modify `nodes`, `vad_dir`, `vad_config`,
# and `nj` when adapting to new clusters.
# ---------------------------------------------------------------------------

#
. ./cmd.sh
. ./path.sh
set -e
nodes=fs01                        # host list when creating split dirs (CLSP grid)
vad_dir=`pwd`/exp/vad_e           # directory to store intermediate VAD artifacts
vad_config=conf/vad_16k.yaml      # path to energy-VAD configuration YAML
nj=40                             # number of parallel jobs used for VAD computation

stage=1                           # stage threshold, raise to skip completed steps
config_file=default_config.sh     # x-vector/VAD config file to source

. parse_options.sh || exit 1;
. $config_file

if [ -z "$vad_config" ];then
  echo "[run002] VAD disabled in configuration, skipping"
  exit 0
fi

if [ "$do_voxsrc22" == "true" ];then
  extra_data="voxsrc22_dev"
fi


if [ $stage -le 1 ]; then
  echo "[run002][stage1] Creating split dirs for VAD jobs"
  for name in voxceleb2cat_train voxceleb1_test $extra_data
  do
    hyp_utils/create_data_split_dirs.sh \
      $vad_dir/$name \
      $USER/hyp-data/voxceleb/v1.2/vad $nodes
  done
fi

#Train datasets
if [ $stage -le 2 ];then
  for name in voxceleb2cat_train voxceleb1_test $extra_data
  do
    # This creates links to distribute data in CLSP grid
    hyp_utils/create_data_split_links.sh $vad_dir/$name/vad.JOB.ark $nj
    echo "[run002][stage2] Computing VAD for $name"
    $train_cmd --array JOB=1:$nj --output-file $vad_dir/$name/log/vad.JOB.log -- \
	       hyperion-compute-energy-vad --cfg $vad_config \
	       --recordings-file data/$name/recordings.csv \
	       --output-spec ark,csv:$vad_dir/$name/vad.JOB.ark,$vad_dir/$name/vad.JOB.csv \
	       --part-idx JOB --num-parts $nj || exit 1

    echo "[run002][stage2] Concatenating per-job VAD tables for $name"
    hyperion-tables cat \
	    --table-type vads \
	    --output-file $vad_dir/$name/vad.csv --num-tables $nj
    echo "[run002][stage2] Registering VAD table with dataset $name"
    hyperion-dataset add_vads \
		  --dataset data/$name \
		  --features-name vad \
		  --features-file $vad_dir/$name/vad.csv
  done
fi
