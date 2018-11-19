#!/bin/bash

TARGET=$HOME/usr/local/anaconda3.5

# Set env vars
export PATH="$TARGET/bin:$PATH"
export PYTHONPATH=""

if [ $# -ne 2 ];then
    echo "Install tf for cpu: $0 1.7 cpu"
    echo "Install tf for gpu: $0 1.7 gpu "
    exit
fi

VERS=$1
DEV=$2

ENV=tensorflow${VERS}g_${DEV}

#Install tensorflow inside a conda environment
#create conda enviroment
conda create -n $ENV python=3.5
#activate enviroment
source activate $ENV

conda install anaconda
conda update pip

if [ "$DEV" == "cpu" ];then
    if [ "$VERS" == "1.8" ];then
	NAME=tensorflow-$VERS.0
    else
	NAME=tensorflow-$VERS.0rc1
    fi
else
    if [ "$VERS" == "1.8" ];then
	NAME=tensorflow_gpu-$VERS.0
    else
	NAME=tensorflow_gpu-$VERS.0rc1
    fi

    export CUDA_TOOLKIT_PATH=/opt/NVIDIA/cuda-9.0
    export CUDNN_INSTALL_PATH=$HOME/usr/local/cudnn-9.0-v7.1
    if [ ! -d $CUDA_TOOLKIT_PATH ];then
	echo "$CUDA_TOOLKIT_PATH not found"
	exit 1
    fi
    if [ ! -d $CUDNN_INSTALL_PATH ];then
	echo "$CUDNN_INSTALL_PATH not found"
	exit 1
    fi
fi

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/$DEV/$NAME-cp35-cp35m-linux_x86_64.whl
pip install $TF_BINARY_URL

#deactivate enviroment
source deactivate $ENV

