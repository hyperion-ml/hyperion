#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh
cal_set=sre19
ncoh=1000

. parse_options.sh || exit 1;
. $config_file

#Video table
args="--print_header true"

be_name=lda200_splday150_v1d_voxceleb_combined_15-800s_40-80utts
score_dir=exp/scores/$nnet_name/${be_name}

dirs=(plda_snorm_v3_ncoh${ncoh}_${diar_name}_ext_cal_v1_${cal_set})
cases=("s-norm v3 ${diar_name} ext")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    score_dir_i=$score_dir/${d}
    local/make_table_line_vid.sh $args "${nnet_name} ${be_name} ${cases[$i]}" $score_dir_i
    args=""
done
