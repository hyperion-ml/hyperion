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

be_name=lda200_splday150_v1_voxceleb_combined
score_dir=exp/scores/$nnet_name/${be_name}

#Video table
args="--print_header true"
dirs=(plda plda_snorm_v1)
cases=("w/o diar" "s-norm w/o diar")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    score_dir_i=$score_dir/${d}
    local/make_table_line_vid.sh $args "${nnet_name} ${be_name} ${cases[$i]}" $score_dir_i
    args=""
done


be_name=lda200_splday150_v1d_voxceleb_combined
score_dir=exp/scores/$nnet_name/${be_name}

dirs=(plda_${diar_name} plda_snorm_v1_${diar_name})
cases=("${diar_name}" "s-norm ${diar_name}")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    score_dir_i=$score_dir/${d}
    local/make_table_line_vid.sh $args "${nnet_name} ${be_name} ${cases[$i]}" $score_dir_i
    args=""
done
