
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools

. $TOOLS_ROOT/path.sh

if [ "$(hostname --domain)" == "cm.gemini" ];then
    module load ffmpeg
    module load cuda10.1/toolkit/10.1.105
    module load cudnn/7.6.3_cuda10.1
else
  CUDA_ROOT=/usr/local/cuda
  CUDA_ROOT=/opt/NVIDIA/cuda-9.1
  LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
  LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH
  if [ ! -d $CUDA_ROOT/lib64 ]; then
    LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
  fi

  #CuDNN env
  CUDNN_ROOT=$TOOLS_ROOT/cudnn/cudnn-10.1-v7.6
  LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH
  LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH
  CPATH=$CUDNN_ROOT/include:$CPATH
fi


export PATH
export LD_LIBRARY_PATH
export LIBRARY_PATH
export CPATH
