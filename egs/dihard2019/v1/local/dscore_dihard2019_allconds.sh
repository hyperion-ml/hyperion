#!/bin/bash
# score all Dihard19 conditions using dscore tool
. path.sh
set -e

if [ $# -ne 3 ];then
    echo "Usage: $0 <dihard-dir> <out-rttm> <output-dir>"
fi

data_dir=$1
rttm=$2
output_dir=$3

mkdir -p $output_dir

conds="all audiobooks broadcast_interview child clinical court maptask meeting restaurant socio_field socio_lab webvideo"
echo "$0  rttm:$rttm"
for cond in $conds
do
    echo "$0 condition: $cond"
    local/dscore_dihard.sh $data_dir $rttm $cond 2> $output_dir/${cond}_err | tee $output_dir/${cond}_results | grep -e "OVERALL" -e File -e "----"
done
