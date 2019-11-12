
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools
export KALDI_ROOT=$TOOLS_ROOT/kaldi/kaldi
PATH=$PWD/utils:$PATH

export LC_ALL=C


#Anaconda env
CONDA_ROOT=$TOOLS_ROOT/anaconda/anaconda3.5
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    #for conda version >=4.4 do    
    . $CONDA_ROOT/etc/profile.d/conda.sh
    conda activate
else
    #for conda version <4.4 do 
    PATH=$CONDA_ROOT/bin:$PATH
fi


if [ "$(hostname --domain)" == "cm.gemini" ];then
    #CUDA env
    module load cuda10.0/toolkit

    #CuDNN env
    CUDNN_ROOT=$TOOLS_ROOT/cudnn/cudnn-10.0-v7.4

    #torch env
    export TORCH=pytorch1.0_cuda10.0

else
    #CUDA env
    export CUDA_ROOT=/usr/local/cuda
    LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUDA_ROOT/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
    CPATH=$CUDA_ROOT/include:$CPATH
    PATH=$CUDA_ROOT/bin:$PATH

    #CuDNN env
    CUDNN_ROOT=$TOOLS_ROOT/cudnn/cudnn-9.0-v7.4

    #torch env
    export TORCH=pytorch1.0_cuda9.0
fi

#CuDNN
LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH
CPATH=$CUDNN_ROOT/include:$CPATH


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
