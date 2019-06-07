
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools

#Kaldi env
export KALDI_ROOT=$TOOLS_ROOT/kaldi/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

#Anaconda env
CONDA_ROOT=$TOOLS_ROOT/anaconda/anaconda3.5
if [ -f "CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    #for conda version >=4.4 do    
    . $CONDA_ROOT/etc/profile.d/conda.sh
    conda activate
else
    #for conda version <4.4 do 
    PATH=$CONDA_ROOT/bin:$PATH
fi

#CUDA env
export CUDA_ROOT=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUDA_ROOT/extras/CUPTI/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
CPATH=$CUDA_ROOT/include:$CPATH
PATH=$CUDA_ROOT/bin:$PATH


#CuDNN env
CUDNN_ROOT=$TOOLS_ROOT/cudnn
export CUDNNV5=$CUDNN_ROOT/cudnn-v5.1
#export CUDNN8V6=$CUDNN_ROOT/cudnn-8.0-v6.0
#export CUDNN8V7=$CUDNN_ROOT/cudnn-8.0-v7.0
export CUDNN9V7=$CUDNN_ROOT/cudnn-9.1-v7.1

#keras back-end
export KERAS_BACKEND=tensorflow
#export KERAS_BACKEND=theano

if [ "$KERAS_BACKEND" == "theano" ];then
    LD_LIBRARY_PATH=$CUDNNV5/lib64:$LD_LIBRARY_PATH
    LIBRARY_PATH=$CUDNNV5/lib64:$LIBRARY_PATH
    CPATH=$CUDNNV5/include:$CPATH
else
    LD_LIBRARY_PATH=$CUDNN9V7/lib64:$LD_LIBRARY_PATH
    LIBRARY_PATH=$CUDNN9V7/lib64:$LIBRARY_PATH
    CPATH=$CUDNN9V7/include:$CPATH
    export TFGPU=tensorflow1.7_gpu
    #export TFCPU=tensorflow1.7_cpu_nomkl
    #export TFGPU=tensorflow1.8g_gpu
    export TFCPU=tensorflow1.8g_cpu
fi


# Keras env
KERAS_PATH=$TOOLS_ROOT/keras
PYTHONPATH=$KERAS_PATH/keras:$KERAS_PATH/keras-applications:$KERAS_PATH/keras-preprocessing:$PYTHONPATH

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
