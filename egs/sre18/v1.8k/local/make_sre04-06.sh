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
for ldc_id in LDC2006S44 LDC2011S01 LDC2011S04 LDC2011S09 LDC2011S10 LDC2012S01
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

    echo "$0 preparing sre04"
    local/make_sre04-12_subset.sh $data_root/LDC2006S44 \
				  04 $master_key $fs $data_dir/sre04
    
    
    echo "$0 preparing sre05"
    local/make_sre04-12_subset.sh $data_root/LDC2011S01 \
				  05 $master_key $fs $data_dir/sre05_train
    
    local/make_sre04-12_subset.sh $data_root/LDC2011S04 \
				  05 $master_key $fs $data_dir/sre05_test
    
    utils/combine_data.sh --extra-files utt2info $data_dir/sre05 \
			  $data_dir/sre05_train $data_dir/sre05_test
    
    echo "$0 preparing sre06"
    local/make_sre04-12_subset.sh $data_root/LDC2011S09 \
				  06 $master_key $fs $data_dir/sre06_train
    
    local/make_sre04-12_subset.sh $data_root/LDC2011S10 \
				  06 $master_key $fs $data_dir/sre06_test_1
    
    local/make_sre04-12_subset.sh $data_root/LDC2012S01 \
				  06 $master_key $fs $data_dir/sre06_test_2
    
    utils/combine_data.sh --extra-files utt2info $data_dir/sre06 \
			  $data_dir/sre06_train $data_dir/sre06_test_1 $data_dir/sre06_test_2
    
elif [ $exist_ldc2009 -eq 1 ];then
    #####################################
    # This is to prepare from LDC2009E100
    #####################################

    data_root=$data_root/LDC2009E100
    
    echo "$0 preparing sre04"
    local/make_sre04-12_subset.sh $data_root/SRE04 \
				  04 $master_key $fs $data_dir/sre04
    
    
    echo "$0 preparing sre05"
    local/make_sre04-12_subset.sh $data_root/SRE05 \
				  05 $master_key $fs $data_dir/sre05
    
    echo "$0 preparing sre06"
    local/make_sre04-12_subset.sh $data_root/SRE06 \
				  06 $master_key $fs $data_dir/sre06

fi

echo "$0 combining sre04-06"
utils/combine_data.sh --extra-files utt2info $data_dir/sre04-06 \
		      $data_dir/sre04 $data_dir/sre05 $data_dir/sre06

