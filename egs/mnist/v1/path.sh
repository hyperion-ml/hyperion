
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools
export KALDI_ROOT=$TOOLS_ROOT/kaldi/kaldi
PATH=$PWD/utils:$PATH

export LC_ALL=C


#Anaconda env
PYTHON_ROOT=$TOOLS_ROOT/anaconda/anaconda3.5
PATH=$PYTHON_ROOT/bin:$PATH

#CUDA env
export CUDA_ROOT=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUDA_ROOT/extras/CUPTI/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
CPATH=$CUDA_ROOT/include:$CPATH
PATH=$CUDA_ROOT/bin:$PATH


#CuDNN env
CUDNN_ROOT=$TOOLS_ROOT/cudnn
export CUDNN9V7=$CUDNN_ROOT/cudnn-9.0-v7.4

LD_LIBRARY_PATH=$CUDNN9V7/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH=$CUDNN9V7/lib64:$LIBRARY_PATH
CPATH=$CUDNN9V7/include:$CPATH

export TORCH=pytorch1.0_cuda9.0

# Matplotlib back-end
export MPLBACKEND="agg"

# Hyp env
export PATH=$HYP_ROOT/hyperion/bin:$PATH
export PYTHONPATH=$HYP_ROOT:$PYTHONPATH
export LIBRARY_PATH
export LD_LIBRARY_PATH
export LC_ALL=C

wait_file() {
    local file="$1"; shift
    local wait_seconds="${2:-30}"; shift # 10 seconds as default timeout
    for((i=0; i<$wait_seconds; i++)); do
	[ -f $file ] && return 1
	sleep 1s
    done
    return 0
}

export -f wait_file
