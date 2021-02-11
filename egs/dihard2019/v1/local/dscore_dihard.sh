#!/bin/bash
# score Dihard using dscore tool
. path.sh
set -e

if [ $# -ne 3 ];then
    echo "Usage: $0 <dihard-dir> <out-rttm> <uem-condition-name>"
fi

data_dir=$1
rttm=$2
uem_name=$3


python $TOOLS_ROOT/dscore/score.py \
    -u $data_dir/uem/$uem_name.uem \
    -r $data_dir/rttm/*.rttm \
    -s $rttm

