#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

print_header=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 <name> <score-dir>"
  exit 1;
fi

name=$1
score_dir=$2


if [ "$print_header" == "true" ];then
    dbs3=("SRE19 DEV" "SRE19 EVAL")
    conds3=(AV)
    dbs4=("JANUS DEV" "JANUS EVAL")
    conds4=(CORE)
    n_dbs3=${#dbs3[*]}
    n_dbs4=${#dbs4[*]}
    n_c1=${#conds1[*]}
    n_c2=${#conds2[*]}
    n_c3=${#conds3[*]}
    n_c4=${#conds4[*]}
    measures=",EER,Min Cp,Act Cp"
    
    printf "System"
    for((i=0;i<$n_dbs3;i++))
    do
	for((j=0;j<$n_c3;j++))
	do
	    printf ",${dbs3[$i]} ${conds3[$j]},,"
	done
    done
    for((i=0;i<$n_dbs4;i++))
    do
	for((j=0;j<$n_c4;j++))
	do
	    printf ",${dbs4[$i]} ${conds4[$j]},,"
	done
    done


    printf "\n"
    nc=$(echo $n_dbs3*$n_c3 + $n_dbs4*$n_c4| bc)
    for((c=0;c<$nc;c++))
    do
	printf "$measures"
    done
    printf "\n"
    
fi


printf "$name,"

for db in dev eval
do
    res_file=$score_dir/sre19_av_v_${db}_results
    awk  '{ printf "%.2f,%.3f,%.3f,", $2,$4,$6}' $res_file
done

for db in dev eval
do
    for c in core
    do
	res_file=$score_dir/janus_${db}_${c}_results
	awk  '{ printf "%.2f,%.3f,%.3f,", $2,$4,$6}' $res_file
    done
done


printf "\n"
