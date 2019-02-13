#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

print_header=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 <name> <sitw-score-dir> <sre18-score-dir>"
  exit 1;
fi

name=$1
sitw_dir=$2
sre18_dir=$3


if [ "$print_header" == "true" ];then

    dbs1=("SITW DEV" "SITW EVAL")
    conds1=(CORE CORE-MULTI)
    dbs2=("SRE18 DEV" "SRE18 EVAL")
    conds2=(VAST)
    n_dbs1=${#dbs1[*]}
    n_dbs2=${#dbs2[*]}
    n_c1=${#conds1[*]}
    n_c2=${#conds2[*]}
    measures=",EER,Min Cp,Act Cp"
    
    printf "System"
    for((i=0;i<$n_dbs1;i++))
    do
	for((j=0;j<$n_c1;j++))
	do
	    printf ",${dbs1[$i]} ${conds1[$j]},,"
	done
    done
    for((i=0;i<$n_dbs2;i++))
    do
	for((j=0;j<$n_c2;j++))
	do
	    printf ",${dbs2[$i]} ${conds2[$j]},,"
	done
    done
    printf "\n"
    nc=$(echo $n_dbs1*$n_c1 + $n_dbs2*$n_c2  | bc)
    for((c=0;c<$nc;c++))
    do
	printf "$measures"
    done
    printf "\n"
    
fi


printf "$name,"

for db in dev eval
do
    for c in core-core core-multi
    do
	res_file=$sitw_dir/sitw_${db}_${c}_results
	awk  '{ printf "%.2f,%.3f,%.3f,", $2,$4,$6}' $res_file
    done
done

for db in dev eval
do
    res_file=$sre18_dir/sre18_${db}_results
    awk  '/VAST/{ printf "%.2f,%.3f,%.3f,", $2,$3,$4}' $res_file

done
printf "\n"
