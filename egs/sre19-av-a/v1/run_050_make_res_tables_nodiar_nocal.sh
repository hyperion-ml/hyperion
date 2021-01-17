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

# nnets=(ftdnn9m.sre18 ftdnn10m.sre18 ftdnn11m.sre18 ftdnn17m.sre18 2a.1.vcc0x5 3a.1.vcc0x5 3d.1.vcc 3d.1.vcc.1-2s 4a.1.vcc 4a.1.vcc.lowsnr)
nnets=(3a.1.vcc 4a.1.vcc 5a.1.vcc)



#Video table
args="--print_header true"
#dirs=(plda plda_${diar_name} plda_snorm plda_${diar_name}_snorm)
#cases=("w/o diar" "${diar_name}" "s-norm w/o diar" "s-norm ${diar_name}")

nc=${#nnets[*]}
for((i=0;i<$nc;i++))
do
    d=${nnets[$i]}
    score_dir=exp/scores/$d/${be_name}/plda
    local/make_table_line_vid.sh $args "$d ${be_name}" $score_dir
    args=""
done

echo ""
