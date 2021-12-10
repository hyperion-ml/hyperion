#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads NIST scoring tools for SRE21

set -e 
tool=sre21_scoring_software
s_dir=sre21/scoring_software

# shareable link:
# https://drive.google.com/file/d/1KyG9Lrl2TnO_iuwwHSlpN45CXK8G2Myn/view?usp=sharing

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1KyG9Lrl2TnO_iuwwHSlpN45CXK8G2Myn" -O $tool.tgz
tar xzvf $tool.tgz

if [ ! -f $s_dir/sre_scorer.py ];then
    echo "the scoring tool wasn't dowloaded correctly, download manually"
    exit 1
fi

rm -f $tool.tgz
