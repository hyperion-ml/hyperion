#!/bin/bash

export KERAS_BACKEND=theano

KERAS_PATH=$HOME/usr/src/keras/keras
HYP_PATH=$(readlink -f ../../../)

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/usr/local/cudnn-v5.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LIBRARY_PATH
export PYTHONPATH=$HYP_PATH:$KERAS_PATH:$PYTHONPATH
export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,optimizer=fast_run,dnn.enabled=True,allow_gc=False,warn_float64=raise"
#export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,optimizer=fast_run,dnn.enabled=False,allow_gc=False,warn_float64=raise"
export MPLBACKEND="agg"

python vae_mnist.py


