#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
if [ $# -ne 4 ]; then
  echo "Usage: $0 <data-root> <master-key> <f_sample> <data-dir>"
  echo "e.g.: $0 /export/corpora/LDC master_key/NIST_SRE_segments_key.csv 8 data/"
  exit 1;
fi


set -e

data_root=$1
master_key=$2
fs=$3
data_dir=$4

#########################################
# Detect the LDC distributions available
#########################################

exist_ldc2009=1
if [ ! -d $data_root/LDC2009E100 ];then
    exist_ldc2009=0
fi
exist_ldc2011=1
for ldc_id in LDC2011S05 LDC2011S07 LDC2011S08
do
    if [ ! -d $data_root/$ldc_id ];then
	exist_ldc2011=0
    fi
done

if [[ $exist_ldc2009 -eq 0 ]] && [[ $exist_ldc2011 -eq 0 ]];then
    echo "Not all the required LDC catalog numbers exists"
    exit 1
fi

if [ $exist_ldc2011 -eq 1 ];then

    ###############################################
    # This is to prepare from LDC2006,2011 releases
    ###############################################

    echo "$0 preparing sre08 train short tel"
    local/make_sre04-12_subset.sh --channel tel $data_root/LDC2011S05 \
				  08 $master_key $fs $data_dir/sre08_train_tel

    echo "$0 preparing sre08 train short interview"
    local/make_sre04-12_subset.sh --channel mic --style interview $data_root/LDC2011S05 \
				  08 $master_key $fs $data_dir/sre08_train_int

    # echo "$0 preparing sre08 train1 mic phonecall"
    # local/make_sre04-12_subset.sh --channel mic --style phonecall $data_root/LDC2011S05 \
    # 				  08 $master_key $fs $data_dir/sre08_train_1_phnmic

    # echo "$0 preparing sre08 train2 tel"
    # local/make_sre04-12_subset.sh --channel tel $data_root/LDC2011S07 \
    # 				  08 $master_key $fs $data_dir/sre08_train_2_tel

    echo "$0 preparing sre08 train long interview"
    local/make_sre04-12_subset.sh --channel mic --style interview --dur long $data_root/LDC2011S07 \
				  08 $master_key $fs $data_dir/sre08_int_long

    # echo "$0 preparing sre08 train2 mic phonecall"
    # local/make_sre04-12_subset.sh --channel mic --style phonecall $data_root/LDC2011S07 \
    # 				  08 $master_key $fs $data_dir/sre08_train_2_phnmic

    echo "$0 preparing sre08 test tel"
    local/make_sre04-12_subset.sh --channel tel $data_root/LDC2011S08 \
				  08 $master_key $fs $data_dir/sre08_test_tel

    echo "$0 preparing sre08 train2 interview"
    local/make_sre04-12_subset.sh --channel mic --style interview $data_root/LDC2011S08 \
				  08 $master_key $fs $data_dir/sre08_test_int

    echo "$0 preparing sre08 train2 mic phonecall"
    local/make_sre04-12_subset.sh --channel mic --style phonecall $data_root/LDC2011S08 \
				  08 $master_key $fs $data_dir/sre08_phnmic

    
    utils/combine_data.sh --extra-files utt2info $data_dir/sre08_tel \
			  $data_dir/sre08_train_tel $data_dir/sre08_test_tel
    utils/combine_data.sh --extra-files utt2info $data_dir/sre08_int \
			  $data_dir/sre08_train_int  $data_dir/sre08_test_int
    #utils/combine_data.sh --extra-files utt2info $data_dir/sre08_phnmic $data_dir/sre08_test_phnmic


elif [ $exist_ldc2009 -eq 1 ];then
    #####################################
    # This is to prepare from LDC2009E100
    #####################################

    data_root=$data_root/LDC2009E100
    
    echo "$0 preparing sre08 tel"
    local/make_sre04-12_subset.sh $data_root/SRE08 \
				  08 $master_key $fs $data_dir/sre08_tel

    echo "$0 preparing sre08 interview"
    local/make_sre04-12_subset.sh --channel mic --style interview $data_root/SRE08 \
				  08 $master_key $fs $data_dir/sre08_int

    echo "$0 preparing sre08 interview long"
    local/make_sre04-12_subset.sh --channel mic --style interview --dur long $data_root/SRE08 \
				  08 $master_key $fs $data_dir/sre08_int_long

    
    echo "$0 preparing sre08 mic phonecall"
    local/make_sre04-12_subset.sh --channel mic --style phonecall $data_root/SRE08 \
				  08 $master_key $fs $data_dir/sre08_phnmic
    
fi


