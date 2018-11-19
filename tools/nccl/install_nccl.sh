#!/bin/bash

#Get nccl_2.1.15-1+cuda9.1_x86_64.txz from NVIDIA
TAR_NAME=nccl_2.1.15-1+cuda9.1_x86_64
DIR_NAME=nccl_2.1.15_cuda9.1

tar xvf $TAR_NAME.txz
cp -r $TAR_NAME $HOME/usr/local/$DIR_NAME
