#!/bin/bash

# Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Create a link to your kaldi installation
# ln -s your-kaldi-path kaldi
#
# Or make new kaldi install from
# kaldi official: https://github.com/kaldi-asr/kaldi.git
# Or from my fork: https://github.com/jesus-villalba/kaldi.git (recommended for some i-vector recipes)
# 

git clone https://github.com/jesus-villalba/kaldi.git

# Install kaldi tools following the instrucctions in kaldi/tools/INSTALL 

cd kaldi/tools
extras/check_dependencies.sh
make -j 16
cd -

# Install kaldi following the instrucctions in kaldi/src/INSTALL 

cd kaldi/src
./configure

# Change -g flag to -O3 in kaldi.mk
sed 's@-g @-O3 @' kaldi.mk > kaldi.mk.tmp
mv  kaldi.mk.tmp  kaldi.mk

make depend -j 16
make -j 16

