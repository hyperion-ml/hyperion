#!/bin/bash

ENV=pytorch1.4_tf2_cuda10.1

#Install tensorflow inside a conda environment
#create conda enviroment
conda create -n $ENV --clone root
#activate enviroment
conda activate $ENV

pip install tensorflow
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
#pip install keras # needed for ibm art toolkit

#deactivate enviroment
conda deactivate $ENV
