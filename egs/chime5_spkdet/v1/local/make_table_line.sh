#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

print_header=false
act_dcf=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 <name> <score-dir>"
  exit 1;
fi

name=$1
dir=$2

conds=(All BIN.SUM U01.CH1 U02.CH1 U04.CH1 U06.CH1)
nc=${#conds[*]}

if [ "$print_header" == "true" ];then

    measures=",EER,DCF(5e-2),DCF(1e-2),DCF(1e-3)"
    
    printf "System"
    for((j=0;j<$nc;j++))
    do
	printf ",${conds[$j]},,,"
    done
    printf "\n"
    for((c=0;c<$nc;c++))
    do
	printf "$measures"
    done
    printf "\n"
fi


printf "$name,"

for((i=0;i<$nc;i++))
do
    c_i=${conds[$i]}
    if [ "$c_i" == "All" ];then
	res_file=$dir/chime5_spkdet_results
    else
	res_file=$dir/chime5_spkdet_${c_i}_results
    fi
    if [ "$act_dcf" == "true" ];then
	awk  '{ printf "%.2f,%.3f/%.3f,%.3f/%.3f,%.3f/%.3f,", $2,$4,$6,$8,$10,$16,$18}' $res_file
    else
	awk  '{ printf "%.2f,%.3f,%.3f,%.3f,", $2,$4,$8,$16}' $res_file
    fi
done

printf "\n"
