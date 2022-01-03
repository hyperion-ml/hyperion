#!/bin/bash
# Copyright 2021 JHU (Author: Jesus Villalba)
# Apache 2.0
#
# Install sph2pipe tool
# Based on Kaldi's makefile
set -e

echo "Installing sph2pipe  tool"

CC=gcc        # used for sph2pipe
WGET=wget
SPH2PIPE_VERSION=v2.5

$WGET -T 10 -t 3 --no-check-certificate https://www.openslr.org/resources/3/sph2pipe_${SPH2PIPE_VERSION}.tar.gz || \
  $WGET -T 10 -c --no-check-certificate https://sourceforge.net/projects/kaldi/files/sph2pipe_${SPH2PIPE_VERSION}.tar.gz; \
  
tar --no-same-owner -xzf sph2pipe_${SPH2PIPE_VERSION}.tar.gz
cd sph2pipe_$SPH2PIPE_VERSION/ && \
  $CC -o sph2pipe  *.c -lm

