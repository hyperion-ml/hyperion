#!/bin/bash

EXP=$1
LABEL="$2"

export KERAS_BACKEND=theano

KERAS_PATH=$HOME/usr/src/keras/keras
HYP_PATH=$(readlink -f ../../../)

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/usr/local/cudnn-v5.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LIBRARY_PATH
export PYTHONPATH=$PWD:$HYP_PATH:$KERAS_PATH:$PYTHONPATH
export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,optimizer=fast_run,dnn.enabled=True,allow_gc=False,warn_float64=raise"
#export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,optimizer=fast_run,dnn.enabled=False,allow_gc=False,warn_float64=raise"
#export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,optimizer=None,exception_verbosity=high,warn_float64=raise"
export MPLBACKEND="agg"


cd $EXP
if [ ! -f data.h5 ];then
    generate_data.py
fi
cd -

vae.py --exp $EXP
cvae.py --exp $EXP
tied_vae_qyqz.py --exp $EXP
tied_vae_qyqzgy.py --exp $EXP
tied_cvae_qyqz.py --exp $EXP
tied_cvae_qyqzgy.py --exp $EXP


plot_val_loss.py --title "$LABEL" \
		 --in-files $EXP/vae_hist.h5 $EXP/cvae_hist.h5 \
		 $EXP/tied_vae_qyqz_hist.h5 $EXP/tied_vae_qyqzgy_hist.h5 \
		 $EXP/tied_cvae_qyqz_hist.h5 $EXP/tied_cvae_qyqzgy_hist.h5 \
		 --out-file $EXP/val_loss.pdf



