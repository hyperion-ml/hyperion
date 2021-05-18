# Basic path.sh file to be used in the hyperon recipes

export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools

export KALDI_ROOT=$TOOLS_ROOT/kaldi/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

HYP_ENV=__HYP_ENV__

#Anaconda env
CONDA_ROOT=$TOOLS_ROOT/anaconda/anaconda3
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    . $CONDA_ROOT/etc/profile.d/conda.sh
    conda activate $HYP_ENV
else
    echo "Anaconda not found in $CONDA_ROOT"
    exit 1
fi

if [ "$(hostname --domain)" == "cm.gemini" ];then
    module load ffmpeg
    module load cuda10.1/toolkit/10.1.105
    module load cudnn/7.6.3_cuda10.1
else
    CUDA_ROOT=/usr/local/cuda
    LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH
    if [ ! -d $CUDA_ROOT/lib64 ]; then
	LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
    fi
fi

export LRU_CACHE_CAPACITY=1 #this will avoid crazy ram memory when using pytorch with cpu, it controls cache of MKLDNN
export HDF5_USE_FILE_LOCKING=FALSE

export MPLBACKEND="agg"
export PATH=$HYP_ROOT/hyperion/bin:$CUDA_ROOT/bin:$PATH
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
