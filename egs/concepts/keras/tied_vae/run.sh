#!/bin/bash

export KERAS_BACKEND="tensorflow"

#CUDA
export CUDA_BASE=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_BASE/lib64:$CUDA_BASE/extras/CUPTI/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH=$CUDA_BASE/lib64:$LIBRARY_PATH
CPATH=$CUDA_BASE/include:$CPATH
export CUDA_ROOT=$CUDA_BASE
export CUDA_HOME=$CUDA_BASE

export CUDNN=$HOME/usr/local/cudnn-9.1-v7.1

export LD_LIBRARY_PATH=$CUDNN/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDNN/lib64:$LIBRARY_PATH
export CPATH=$CUDNN/include:$CPATH

source activate tensorflow1.7_cpu_nomkl

run_tsvae.py

source deactivate tensorflow1.7_cpu_nomkl
