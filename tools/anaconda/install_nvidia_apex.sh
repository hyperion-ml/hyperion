#!/bin/bash

#Install NVIDIA Apex to be able to use mixed precision training
# you need a recent gcc compiler
# in COE grid it works with  
# module load gcc/5.4.0

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
