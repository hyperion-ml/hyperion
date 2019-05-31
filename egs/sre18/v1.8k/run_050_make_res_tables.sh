#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

score_dir=exp/scores/$nnet_name/tel_${be_tel_name}_vid_${be_vid_name}

#Video table
args="--print_header true"
dirs=(plda plda_${diar_name} plda_snorm plda_${diar_name}_snorm)
cases=("w/o diar" "${diar_name}" "s-norm w/o diar" "s-norm ${diar_name}")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    dir_sitw_i=$score_dir/${d}_cal_v1
    dir_sre18_i=$score_dir/sre18_${d}_cal_v1
    local/make_table_line_vid.sh $args "${nnet_name} ${cases[$i]}" $dir_sitw_i $dir_sre18_i
    args=""
done

echo ""

#Tel table
args="--print_header true"
dirs=(plda plda_snorm)
cases=("" "s-norm")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    dir_sre18_i=$score_dir/sre18_${d}_cal_v1
    local/make_table_line_tel.sh $args "${nnet_name} ${cases[$i]}" $dir_sre18_i
    args=""
done


