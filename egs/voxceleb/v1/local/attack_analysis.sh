#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

cmd=run.pl
prior=0.05
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 <key> <clean-scores> <adv-scores-array> <adv-stats-array> <output-basename>"
  exit 1;
fi

set -e

key=$1
clean_scores=$2
adv_scores="$3"
adv_stats="$4"
output_path=$5

output_dir=$(dirname $output_path)
base=$(basename $output_path)
logdir=$output_dir/log
mkdir -p $logdir

if [ "$(hostname --domain)" == "cm.gemini" ];then
    module load texlive
fi

$cmd $logdir/analysis_${base}.log \
    local/attack_analysis.py \
    --key-file $key \
    --clean-score-file $clean_scores \
    --attack-score-files $adv_scores \
    --attack-stats-files $adv_stats \
    --output-path $output_path



