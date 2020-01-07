#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

all_conds=false
print_header=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 <name> <sre18-score-dir>"
  exit 1;
fi

name=$1
sitw_dir=$2
sre18_dir=$3

if [ "$print_header" == "true" ];then

    dbs1=("SRE18 DEV" "SRE18 EVAL")
    if [ "$all_conds" == "false" ];then
	conds1=(CMN2)
    else
	conds2=(CMN2 "CMN2 MALE" "CMN2 FEMALE" "CMN2 PSTN" "CMN2 VOIP" "CMN2 SAMEPHN" "CMN2 DIFFPHN" "CMN2 ENR1" "CMN2 ENR3")
    fi
    n_dbs1=${#dbs1[*]}
    n_c1=${#conds1[*]}
    measures=",EER,Min Cp,Act Cp"
    
    printf "System"
    for((i=0;i<$n_dbs1;i++))
    do
	for((j=0;j<$n_c1;j++))
	do
	    printf ",${dbs1[$i]} ${conds1[$j]},,"
	done
    done
    printf "\n"
    nc=$(echo $n_dbs1*$n_c1 | bc)
    for((c=0;c<$nc;c++))
    do
	printf "$measures"
    done
    printf "\n"
    
fi

name=$1
sre18_dir=$2

conds="male female pstn voip samephn diffphn enroll1 enroll3"
printf "$name,"

for db in dev eval
do
    res_file=$sre18_dir/sre18_${db}_cmn2_results
    awk  '/CMN2/{ printf "%.2f,%.3f,%.3f,", $2,$3,$4}' $res_file
    
    if [ "$all_conds" == "true" ];then 
	for cond in $conds
	do
	    awk  '/CMN2/ { printf "%.2f,%.3f,%.3f,", $2,$3,$4}' ${res_file}_$cond
	done
    fi
done

printf "\n"
