#!/bin/bash

###########################################################################
# Install Anaconda default packages

TARGET=$HOME/usr/local/anaconda3.5

# Set env vars
export PATH="$TARGET/bin:$PATH"
export PYTHONPATH=""


cd $TARGET/bin

#I dont know if we need this to find cudnn5 but just in case
export CPATH=$HOME/usr/local/cudnn-8.0-v6.0/include:$CPATH
export LD_LIBRARY_PATH=$HOME/usr/local/cudnn-8.0-v6.0/lib64:$LD_LIBRARY_PATH


#Install tensorflow inside a conda environment
#create conda enviroment
conda create -n tensorflow1.4_gpu python=3.5
#activate enviroment
source activate tensorflow1.4_gpu

conda install anaconda

#Install tf
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp35-cp35m-linux_x86_64.whl
pip install --ignore-installed --upgrade $TF_BINARY_URL


#deactivate enviroment
source deactivate tensorflow1.4_gpu

#Install tensorflow for cpu inside a conda environment
#create conda enviroment
conda create -n tensorflow1.4_cpu python=3.5
#activate enviroment
source activate tensorflow1.4_cpu

conda install anaconda

#Install tf
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0rc1-cp35-cp35m-linux_x86_64.whl
pip install --ignore-installed --upgrade $TF_BINARY_URL

#deactivate enviroment
source deactivate tensorflow1.4_cpu

cd - 

