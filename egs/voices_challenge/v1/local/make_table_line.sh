#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

print_header=false
cond_type=""

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 <name> <score-dir>"
  exit 1;
fi

name=$1
dir=$2

if [ "$cond_type" == "" ];then
    conds_dev=(All)
    conds_eval=(All)
elif [ "$cond_type" == "orientation" ];then
    conds_dev=(dg000 dg060 dg090 dg120 dg180)
    conds_eval=(dg000 dg030 dg060 dg090 dg120 dg150 dg180)
elif [ "$cond_type" == "noise" ];then
    conds_dev=(none babb musi tele)
    conds_eval=(none babb tele)
elif [ "$cond_type" == "mic_type" ];then
    conds_dev=(lav)
    conds_eval=(lav bar mem)
elif [ "$cond_type" == "mic_pos" ];then
    conds_dev=(beh cec ceo clo far mid tbo wal)
    conds_eval=(beh cab ceh ceo clo far mid ref tab und wal)
fi

nc_dev=${#conds_dev[*]}
nc_eval=${#conds_eval[*]}
nc=$(($nc_dev+$nc_eval))

if [ "$print_header" == "true" ];then

    measures=",EER,MinDCF,ActDCF"
    printf ",VOICES DEV,,"
    for((j=1;j<$nc_dev;j++))
    do
	printf ",,,"
    done
    printf ",VOICES EVAL,,"
    for((j=1;j<$nc_eval;j++))
    do
	printf ",,,"
    done
    printf "\n"
    printf "System"
    for((j=0;j<$nc_dev;j++))
    do
	printf ",${conds_dev[$j]},,"
    done
    for((j=0;j<$nc_eval;j++))
    do
	printf ",${conds_eval[$j]},,"
    done
    printf "\n"
    for((c=0;c<$nc;c++))
    do
	printf "$measures"
    done
    printf "\n"
fi


printf "$name,"

for((i=0;i<$nc_dev;i++))
do
    c_i=${conds_dev[$i]}
    if [ "$c_i" == "All" ];then
	res_file=$dir/voices19_challenge_dev_results
    else
	res_file=$dir/voices19_challenge_dev_results.$c_i
    fi
    awk '$1=="EER" { eer=$3*100}
         $1=="minDCF" { min_dcf=$3}
         $1=="actDCF" { act_dcf=$3}
         END{ printf "%.2f,%.3f,%.3f,", eer,min_dcf, act_dcf }' $res_file
done

for((i=0;i<$nc_eval;i++))
do
    c_i=${conds_eval[$i]}
    if [ "$c_i" == "All" ];then
	res_file=$dir/voices19_challenge_eval_results
    else
	res_file=$dir/voices19_challenge_eval_results.$c_i
    fi
    awk '$1=="EER" { eer=$3*100}
         $1=="minDCF" { min_dcf=$3}
         $1=="actDCF" { act_dcf=$3}
         END{ printf "%.2f,%.3f,%.3f,", eer,min_dcf, act_dcf }' $res_file
done


printf "\n"
