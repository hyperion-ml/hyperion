#!/bin/bash

###########################################################################
# This is just an example of how to install Anaconda

TARGET=$HOME/usr/local/anaconda3
bash Anaconda3-4.2.0-Linux-x86_64.sh -p $TARGET

# This was for old conda versions
# Set env vars
#export PATH="$TARGET/bin:$PATH"
#export PYTHONPATH=""
#Now just make sure that you activate the base environment of the right conda with
conda activate

# Update conda package manager
# 3 times untils there is nothing more
conda update conda
conda update conda
conda update conda

#Update numpy, scipy, ...
conda update anaconda

# Install google protobuffers
conda install protobuf

#update pip
conda update pip

#install sphinx
conda install sphinx
pip install sphinx_rtd_theme

#install soundfile
pip install pysoundfile


#this is needed by theano
#conda install pygpu

###########################################################################
###########################################################################
#Only need it in 2.7
#Copy sitecustomize.py to packages directory
#   Change default encoding to utf-8

#cp sitecustomize.py $TARGET/lib/python2.7/site-packages/

###########################################################################
###########################################################################
# Install extra packages not included in Anaconda
# but that can be installed using pip package manager


# # Install theano
# cd $TARGET/bin

# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# cd -
