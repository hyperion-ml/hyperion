#!/bin/bash
# Copyright
#                2023   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
# Creates links to distrubute data into multiple nodes in clsp grid

if [ $# -ne 3 ]; then
  echo "Usage: $0 <output-dir> <recordings-file> <audio-format>"
  echo "$0 exp/xvector_audios/voxceleb data/voxceleb/recordings.csv flac"
fi
echo "$0 $@"  # Print the command line for logging
output_dir=$1
rec_file=$2
file_format=$3

if [[ $(hostname -f) != *.clsp.jhu.edu ]]; then
   exit 0
fi

for f in $(awk -F "," '$1!="id" { print $1}' $rec_file); do
  # the next command does nothing unless $output_dir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  hyp_utils/create_data_link.pl $output_dir/$f.$file_format
done



