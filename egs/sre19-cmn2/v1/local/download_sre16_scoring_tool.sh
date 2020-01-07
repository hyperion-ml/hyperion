#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads NIST scoring tools for SRE16

set -e 
tool=sre16_scoring_software
s_dir=scoring_software/sre16

# shareable link:
# https://drive.google.com/file/d/1-jO9Y16uASLmV8vdwVeY5HrSY0Tnjr5o/view?usp=sharing

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1-jO9Y16uASLmV8vdwVeY5HrSY0Tnjr5o" -O $tool.tar.bz2
mkdir -p $s_dir
tar xjvf $tool.tar.bz2 -C $s_dir

if [ ! -f $s_dir/scoring.py ];then
    echo "the scoring tool wasn't dowloaded correctly"
    exit 1
fi

rm -f $tool.tar.bz2
