
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools

export KALDI_ROOT=$TOOLS_ROOT/kaldi/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

#Anaconda env
CONDA_ROOT=$TOOLS_ROOT/anaconda/anaconda3.5
CONDA_ROOT=$TOOLS_ROOT/anaconda/anaconda3
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    #for conda version >=4.4 do    
    . $CONDA_ROOT/etc/profile.d/conda.sh
    conda activate
else
    #for conda version <4.4 do 
    PATH=$CONDA_ROOT/bin:$PATH
fi

if [ "$(hostname --domain)" == "cm.gemini" ];then
    module load cuda10.0/toolkit
    module load ffmpeg
    TORCH="pytorch1.0_cuda10.0"
    CUDNN_ROOT=$TOOLS_ROOT/cudnn/cudnn-10.0-v7.4
else
    #CUDA_ROOT=/opt/NVIDIA/cuda-9.0
    CUDA_ROOT=/home/janto/usr/local/cuda-10.1
    LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH
    if [ ! -d $CUDA_ROOT/lib64 ]; then
	LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
    fi
    #TORCH="pytorch1.0_cuda9.0"
    TORCH="pytorch1.4_cuda10.1"
    #CuDNN env
    #CUDNN_ROOT=$TOOLS_ROOT/cudnn/cudnn-9.0-v7.4
    CUDNN_ROOT=$TOOLS_ROOT/cudnn/cudnn-10.1-v7.6
fi
#CuDNN
LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH
CPATH=$CUDNN_ROOT/include:$CPATH

export LRU_CACHE_CAPACITY=1 #this will avoid crazy ram memory when using pytorch with cpu, it controls cache of MKLDNN

export MPLBACKEND="agg"
export PATH=$HYP_ROOT/hyperion/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$HYP_ROOT:$KERAS_PATH:$PYTHONPATH
export LD_LIBRARY_PATH
export LC_ALL=C

export HDF5_USE_FILE_LOCKING=FALSE

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
