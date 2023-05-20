#!/bin/bash
# Copyright
#                2023   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
# Creates links to distrubute data into multiple nodes in clsp grid

storage_name=$(date +'%m_%d_%H_%M')

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 <output-dir> <storage-dir> <nodes>"
  echo "$0 exp/vad_dir $USER/hyp-data/voxceleb/v1/vad/storage b0"
fi
output_dir=$1
storage_dir=$2
nodes=$3

link_dir=$output_dir/storage

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $linkdir ]; then
  echo "Prepare to distribute data over multiple $nodes nodes"
  dir_name=$storage_dir/$storage_name/storage
  if [ "$nodes" == "b0" ];then
    utils/create_split_dir.pl \
      hyp_utils/create_split_dir.pl \
      /export/b{04,05,06,07}/$dir_name $link_dir
  elif [ "$nodes" == "b1" ];then
    hyp_utils/create_split_dir.pl \
      /export/b{14,15,16,17}/$dir_name $link_dir
  elif [ "$nodes" == "c0" ];then
    hyp_utils/create_split_dir.pl \
      /export/c{06,07,08,09}/$dir_name $link_dir
  elif [ "$nodes" == "fs01" ];then
    hyp_utils/create_split_dir.pl \
      /export/fs01/$dir_name $link_dir
  else
    echo "we don't distribute data between multiple machines"
  fi
fi



