#!/bin/bash

TARGET=$PWD/anaconda3.5

# Set env vars
export PATH="$TARGET/bin:$PATH"
export PYTHONPATH=""

ENV=pytorch1.0_cuda9.0

#Install tensorflow inside a conda environment
#create conda enviroment
conda create -n $ENV --clone root
#activate enviroment
source activate $ENV

#export CUDA_TOOLKIT_PATH=/opt/NVIDIA/cuda-9.0

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

#deactivate enviroment
source deactivate $ENV
