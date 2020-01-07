#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads NIST scoring tools for SRE18

set -e 
tool=sre18_scoring_software
s_dir=scoring_software/sre18

# shareable link:
# https://drive.google.com/file/d/1VqXEt8OyNSEBo0QxhPtH1TQkNVO15urQ/view?usp=sharing

mkdir -p scoring_software

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1VqXEt8OyNSEBo0QxhPtH1TQkNVO15urQ" -O $tool.tgz
tar xzvf $tool.tgz -C scoring_software
mv scoring_software/scoring_software $s_dir

if [ ! -f $s_dir/sre_scorer.py ];then
    echo "the scoring tool wasn't dowloaded correctly"
    exit 1
fi

rm -f $tool.tgz
