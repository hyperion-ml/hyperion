#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 4 ]; then
  echo "Usage: $0 <data-root> <master-key> <f_sample> <data-dir>"
  echo "e.g.: $0 /export/corpora/LDC master_key/NIST_SRE_segments_key.csv 8 data/"
  exit 1;
fi


set -e

data_root=$1
master_key=$2
fs=$3
data_dir=$4


echo "$0 preparing sre10 interview"
local/make_sre04-12_subset.sh --orig-fs 16 --channel mic --style interview $data_root \
			      10 $master_key $fs $data_dir/sre10_int


echo "$0 preparing sre10 interview long"
local/make_sre04-12_subset.sh --orig-fs 16 --channel mic --style interview --dur long $data_root \
			      10 $master_key $fs $data_dir/sre10_int_long


echo "$0 preparing sre10 mic phonecall"
local/make_sre04-12_subset.sh --orig-fs 16 --channel mic --style phonecall $data_root \
			      10 $master_key $fs $data_dir/sre10_phnmic




