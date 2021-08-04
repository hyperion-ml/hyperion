#!/bin/bash

. path.sh

echo "Download insightface repo"
# We use fork of repo at https://github.com/deepinsight/insightface.git
git clone https://github.com/jesus-villalba/insightface.git
cd insightface
#use sre21 branch
git checkout sre21-cpt
cd -

echo "#Download deformable conv nets repo"
git clone https://github.com/msracver/Deformable-ConvNets.git

echo "Compile retina face"
conda activate $MXNET_ENV
cd insightface/detection/retinaface
make
cd -
conda deactivate
