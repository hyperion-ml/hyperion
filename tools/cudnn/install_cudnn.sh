#!/bin/bash

echo "
Get cudnn-v7 for your version of cuda toolkit from NVIDIA
Install in $HOME/usr/local and add to LD_LIBRARY_PATH
"

TAR_NAME=cudnn-9.1-linux-x64-v7.1.tgz
DIR_NAME=cudnn-9.1-v7.1

tar xzvf $TAR_NAME.tgz
cp -r $TAR_NAME $HOME/usr/local/$DIR_NAME
