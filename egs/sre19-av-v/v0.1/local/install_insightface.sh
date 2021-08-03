#!/bin/bash

. path.sh

echo "Download insightface repo"
git clone https://github.com/deepinsight/insightface.git

echo "#Download deformable conv nets repo"
git clone https://github.com/msracver/Deformable-ConvNets.git

echo "Compile retina face"
conda activate $MXNET_ENV
cd insightface/detection/retinaface
make
cd -
conda deactivate
