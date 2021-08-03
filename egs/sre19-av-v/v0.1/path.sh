
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools

. $TOOLS_ROOT/path.sh

INSIGHT_ROOT=$PWD/insightface
RETINA_PATH=$INSIGHT_ROOT/detection/retinaface
FACERECO_PATH=$INSIGHT_ROOT/recognition/arcface_mxnet
FACERECO_PATH=$PWD/steps_insightface/deploy

MXNET_ENV=hyperion_tyche
#echo "MXNet Environment=$MXNET_ENV"

if [ "$(hostname --domain)" == "cm.gemini" ];then
  module load cuda10.2/toolkit
  module load cudnn/8.0.2_cuda10.2
  module load nccl/2.7.8_cuda10.2
else
  # Add CUDA paths for your grid
  CUDA_ROOT=/usr/local/cuda
  CUDNN_ROOT=""
  NCCL_ROOT=""
  
  LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUDA_ROOT/lib:$LD_LIBRARY_PATH
  LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$CUDNN_ROOT/lib:$LD_LIBRARY_PATH
  LD_LIBRARY_PATH=$NCCL_ROOT/lib64:$NCCL_ROOT/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH
fi
export PYTHONPATH=$RETINA_PATH:$FACERECO_PATH:$PYTHONPATH

