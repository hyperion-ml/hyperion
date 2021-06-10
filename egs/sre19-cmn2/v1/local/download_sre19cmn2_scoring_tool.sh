#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads NIST scoring tools for SRE19

set -e 
tool=sre19_cts_challenge_scoring_software
s_dir=scoring_software/sre19-cmn2

# shareable link:
# https://drive.google.com/file/d/16c68AOjBlKsAiGVj-iMAWjA4tlSITgn8/view?usp=sharing

mkdir -p scoring_software
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=16c68AOjBlKsAiGVj-iMAWjA4tlSITgn8" -O $tool.tbz2
tar xjvf $tool.tbz2

if [ ! -f $s_dir/sre_scorer.py ];then
    echo "the scoring tool wasn't dowloaded correctly"
    exit 1
fi

rm -f $tool.tbz2
