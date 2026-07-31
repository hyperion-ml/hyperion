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

scores_v=($adv_scores)
for((i=0;i<${#scores_v[@]};i++))
do
    scores_dir=$(dirname ${scores_v[$i]})
    wav_out_dir0=${output_path}_wavs

    for t in tar non
    do
	if [ "$t" == "tar" ];then
	    t2=tar2non
	else
	    t2=non2tar
	fi
	wav_in_dir=$scores_dir/wav/$t2
	if [ ! -d "$wav_in_dir" ];then
	    continue
	fi
	for m in snr linf
	do
	    best_file=${output_path}_best_${m}_${t}_attacks_$i.csv
	    if [ ! -f $best_file ];then
		continue
	    fi
	    wav_out_dir=${wav_out_dir0}/best_${m}_${t}_attacks_$i
	    mkdir -p $wav_out_dir
	    for f in $(awk -F "," 'BEGIN{getline;}{ print $2"-"$3".wav"}' $best_file)
	    do
		ff=$wav_in_dir/$f
		if [ -f $ff ];then
		    cp -v $ff $wav_out_dir > $logdir/copywavs_${base}.log 2>&1
		fi
	    done
	done
    done
done


