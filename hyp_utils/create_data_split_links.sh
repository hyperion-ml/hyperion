#!/bin/bash
# Copyright
#                2023   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
# Creates links to distrubute data into multiple nodes in clsp grid

storage_name=$(date +'%m_%d_%H_%M')

echo "$0 $@"  # Print the command line for logging
if [ $# -ne 3 ]; then
  echo "Usage: $0 <output-file-pattern> < <num-jobs>"
  echo "$0 exp/vad_dir/vad.JOB.ark 40"
fi
output_file_pattern=$1
nj=$2

for n in $(seq $nj); do
  # the next command does nothing unless output_dir/storage exists, see
  # utils/create_data_link.pl for more info.
  output_file=$(echo $output_file_pattern | sed 's@\.JOB\.[^\.]*$@.'$n'.@')
  hyp_utils/create_data_link.pl $output_file
done

