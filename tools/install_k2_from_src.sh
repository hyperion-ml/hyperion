#!/bin/bash

if [ $# -ne 2 ];then
    echo "Usage: $0 <conda-env-name> <cuda-root-path>"
    echo "  e.g.: $0 hyperion /usr/local/cuda"
fi

env_name=$1
CUDA_ROOT=$2

eval "$(conda shell.bash hook)"
conda activate $env_name

#module load cuda10.2/toolkit
#module load gcc

#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

CUDA_VERSION=$(echo "import torch; print(torch.version.cuda)" | python)
CUDNN_VERSION=$(echo "import torch; print(torch.__config__.show())" | python | awk '/CuDNN/ { print $NF}')

# Install cmake
echo "Installing CMAKE"
conda install -c anaconda cmake
echo "Installing NVIDIDA CUDA=$CUDA_VERSION CUDNN=$CUDNN_VERSION"
conda install -c nvidia cudnn=$CUDNN_VERSION cudatoolkit=$CUDA_VERSION

#conda install -c k2-fsa -c conda-forge kaldilm

echo "Download k2"
git clone https://github.com/k2-fsa/k2.git
cd k2

ENV_PATH=$(which python | sed 's@/bin/python$@@')
NVCC=$CUDA_ROOT/bin/nvcc
CUDNN_LIBRARY_PATH=${ENV_PATH}/lib
CUDNN_INCLUDE_PATH=${ENV_PATH}/include
CUDA_TOOLKIT_DIR=$ENV_PATH
export PATH=$CUDA_ROOT/bin:$PATH

export K2_CMAKE_ARGS="\
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CUDA_COMPILER=$NVCC \
-DPYTHON_EXECUTABLE=$(which python) \
-DCUDNN_LIBRARY_PATH=$CUDNN_LIBRARY_PATH/libcudnn.so \
-DCUDNN_INCLUDE_PATH=$CUDNN_INCLUDE_PATH \
-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT"

export K2_MAKE_ARGS="-j6"

echo "Compile k2 with CMAKE_ARGS=$K2_CMAKE_ARGS"
python setup.py install
cd -


# pip install lhotse

# export OT_CMAKE_ARGS=$K2_CMAKE_ARGS
# git clone https://github.com/csukuangfj/optimized_transducer
# cd optimized_transducer
# python setup.py install
# cd -


# git clone https://github.com/k2-fsa/icefall
# cd icefall
# pip install -r requirements.txt
# export PYTHONPATH=./icefall:$PYTHONPATH
