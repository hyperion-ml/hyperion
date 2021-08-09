#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

cal_set=sre19

. parse_options.sh || exit 1;


name=$2
score_dir=$1

#Video table
args="--print_header true"
dirs=(cosine_cal_v1_${cal_set} cosine_snorm1000_v1_cal_v1_${cal_set})
cases=("" "s-norm")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    score_dir_i=$score_dir/${d}
    local/make_table_line_vid.sh $args "${name} ${cases[$i]}" $score_dir_i
    args=""
done


