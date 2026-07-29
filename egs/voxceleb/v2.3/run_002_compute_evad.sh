#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e
nodes=fs01
vad_dir=`pwd`/exp/vad_e
vad_config=conf/vad_16k.yaml
nj=40

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ -z "$vad_config" ];then
  echo "We are not using VAD in this configuration"
  exit 0
fi

if [ "$do_voxsrc22" == "true" ];then
  extra_data="voxsrc22_dev"
fi


if [ $stage -le 1 ]; then
  # Prepare to distribute data over multiple machines
  # This only does something at CLSP grid
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
    # If you are not at CLSP grid, it does nothing and can be deleted
    hyp_utils/create_data_split_links.sh $vad_dir/$name/vad.JOB.ark $nj
    echo "compute vad for $name"
    $train_cmd JOB=1:$nj $vad_dir/$name/log/vad.JOB.log \
	       hyp_utils/conda_env.sh \
	       hyperion-compute-energy-vad --cfg $vad_config \
	       --recordings-file data/$name/recordings.csv \
	       --output-spec ark,csv:$vad_dir/$name/vad.JOB.ark,$vad_dir/$name/vad.JOB.csv \
	       --part-idx JOB --num-parts $nj || exit 1

    hyperion-tables cat \
		    --table-type features \
		    --output-file $vad_dir/$name/vad.csv --num-tables $nj
    hyperion-dataset add_features \
		     --dataset data/$name \
		     --features-name vad \
		     --features-file $vad_dir/$name/vad.csv
  done
fi


