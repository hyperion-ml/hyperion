#!/bin/bash

export KERAS_BACKEND=theano

KERAS_PATH=$HOME/usr/src/keras/keras
#CRONUS_PATH=$HOME/usr/src/cronus/cronus
CRONUS_PATH=$(readlink -f ../../../)

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/usr/local/cudnn-v5.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LIBRARY_PATH
export PYTHONPATH=$CRONUS_PATH:$KERAS_PATH:$PYTHONPATH
export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,optimizer=fast_run,dnn.enabled=True,allow_gc=False,warn_float64=raise"
#export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,optimizer=fast_run,dnn.enabled=False,allow_gc=False,warn_float64=raise"
export MPLBACKEND="agg"

if [ ! -f data.h5 ];then
    python generate_data.py
fi

echo run_plda | matlab -nodisplay

#python tied_vae_ivec.py
python tied_vae_qyqzgy_ivec.py

