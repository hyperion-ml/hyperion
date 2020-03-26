#!/bin/bash

ENV=pytorch1.4_cuda10.1

#Install tensorflow inside a conda environment
#create conda enviroment
conda create -n $ENV --clone root
#activate enviroment
conda activate $ENV

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

#deactivate enviroment
conda deactivate $ENV
