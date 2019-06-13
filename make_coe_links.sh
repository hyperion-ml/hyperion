#!/bin/bash

#This script creates links in the tools directory to
# - kaldi
# - anaconda3.5
# - cudnn
# This avoids that every person need its own copy

#kaldi
cd tools/kaldi
if [ ! -f kaldi ]; then
    ln -s /exp/jvillalba/kaldi/kaldi-villalba kaldi
fi
cd -

# anaconda 3.5
cd tools/anaconda
if [ ! -f anaconda3 ];then
    ln -s /exp/jvillalba/anaconda/anaconda3 anaconda3.5
fi
cd -

# cudnn
cd tools/cudnn
#cudnn v7.4 for cuda 9.0 needed by pytorch 1.0 (conda enviroment pytorch1.0_cuda9.0)
if [ ! -f cudnn-9.0-v7.4 ];then
    ln -s /home/hltcoe/jvillalba/usr/local/cudnn-10.0-v7.4 cudnn-10.0-v7.4 
fi
cd -

#nccl
cd tools/nccl
if [ ! -f nccl_2.4.2_cuda10.0 ];then
    ln -s /home/hltcoe/jvillalba/usr/local/nccl_2.4.2-1+cuda10.0_x86_64 nccl_2.4.2_cuda10.0
fi


#cudnn v7.1 for cuda 9.1 needed by tf1.8 (conda environment tensorflow1.8g_gpu)
#if [ ! -f cudnn-9.1-v7.1 ];then
#    ln -s /home/janto/usr/local/cudnn-9.1-v7.1 cudnn-9.1-v7.1
#fi


