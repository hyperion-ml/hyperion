#!/bin/bash

#This script creates links in the tools directory to
# - kaldi
# - anaconda3
# This avoids that every person need its own copy

#kaldi
cd tools
if [ ! -f kaldi ]; then
    ln -s /export/b15/janto/kaldi/kaldi-villalba kaldi
fi

# anaconda 3
if [ ! -f anaconda3 ];then
    ln -s /home/janto/usr/local/anaconda3 anaconda3
fi
cd -

# # cudnn
# cd tools/cudnn
# #cudnn v7.6 for cuda 10.1 needed by pytorch 1.4 (conda enviroment pytorch1.4_cuda10.1)
# if [ ! -f cudnn-10.1-v7.6 ];then
#     ln -s /home/janto/usr/local/cudnn-10.1-v7.6 cudnn-10.1-v7.6
# fi


# #deprecated from here
# #cudnn v7.4 for cuda 9.0 needed by pytorch 1.0 (conda enviroment pytorch1.0_cuda9.0)
# if [ ! -f cudnn-9.0-v7.4 ];then
#     ln -s /home/janto/usr/local/cudnn-9.0-v7.4 cudnn-9.0-v7.4 
# fi

# #cudnn v7.1 for cuda 9.1 needed by tf1.8 (conda environment tensorflow1.8g_gpu)
# if [ ! -f cudnn-9.1-v7.1 ];then
#     ln -s /home/janto/usr/local/cudnn-9.1-v7.1 cudnn-9.1-v7.1
# fi


